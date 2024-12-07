import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
class TriPlane(nn.Module):
    def __init__(self, model_params):
        super(TriPlane, self).__init__()
        
        self.coarse_plane_res = model_params['coarse']
        self.fine_plane_res = model_params['fine']
        c_dim = model_params['c_dim']
        xyz_min = [model_params['xmin'], model_params['ymin'], model_params['zmin']]
        xyz_max = [model_params['xmax'], model_params['ymax'], model_params['zmax']]
        self.xyz_min = torch.tensor(xyz_min).cuda()
        self.xyz_max = torch.tensor(xyz_max).cuda()
        xyz_len = self.xyz_max - self.xyz_min

        planes_xy, planes_xz, planes_yz = [], [], []
        planes_res = [self.coarse_plane_res, self.fine_plane_res]
        planes_dim = c_dim
        
        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
            print('Plane Shape:', grid_shape)
        self.planes_xy = nn.ParameterList([nn.Parameter(p) for p in planes_xy])
        self.planes_xz = nn.ParameterList([nn.Parameter(p) for p in planes_xz])
        self.planes_yz = nn.ParameterList([nn.Parameter(p) for p in planes_yz])
        
    def forward(self, xyz):
        xyz = xyz.cuda().detach()
        indnorm = (xyz-self.xyz_min)*2.0 / (self.xyz_max-self.xyz_min) -1
        vgrid = indnorm[None, :, None]
        feat = []
        for i in range(len(self.planes_xy)):
            xy = F.grid_sample(self.planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(self.planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(self.planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)
        feat = torch.cat(feat, dim=-1)
        return feat

class Decoder(nn.Module):
    def __init__(self, model_params):
        super(Decoder, self).__init__()
        mlp_width = model_params['mlp_dim']
        output_dim = model_params['out_dim']
        self.single_mlp = nn.Sequential(nn.Linear(model_params['c_dim']*2, mlp_width),
                                        nn.ReLU(),
                                        nn.Linear(mlp_width, mlp_width),
                                        nn.ReLU(),
                                        nn.Linear(mlp_width, output_dim)
                                        )
    def forward(self, feat):
        return self.single_mlp(feat)

class GaussianLearner(nn.Module):
    def __init__(self, model_params, xyz_min = [-2, -2, -2], xyz_max=[2, 2, 2] ):
        super(GaussianLearner, self).__init__()

        self.Q0 = 0.03
        xyz_min = [model_params['xmin'], model_params['ymin'], model_params['zmin']]
        xyz_max = [model_params['xmax'], model_params['ymax'], model_params['zmax']]
        self.xyz_min = torch.tensor(xyz_min).cuda()
        self.xyz_max = torch.tensor(xyz_max).cuda()

        self.world_size = [model_params['coarse'], model_params['fine']]
        self.max_step = 6
        self.current_step = 0

        self._feat = FeaturePlanes(use_single_mlp = model_params['use_single_mlp'], world_size=self.world_size, xyz_min = self.xyz_min, xyz_max= self.xyz_max,
                                    feat_dim = model_params['num_channels'], mlp_width = [model_params['mlp_dim']], out_dim=[8], subplane_multiplier=model_params['subplane_multiplier'] )  # 27,4,3,1

        self.register_buffer('opacity_scale', torch.tensor(10))
        self.opacity_scale = self.opacity_scale.cuda()


    # def activate_plane_level(self):
    #     self._feat.activate_level +=1
    #     print('******* Plane Level to:', self._feat.activate_level)


    def inference(self, xyz):
        inputs = xyz.cuda().detach()
        
        tmp  = self._feat(inputs, self.Q0)
        # features = tmp[:,:27]
        rotations = tmp[:,:4]
        scale = tmp[:,4:4+3]
        opacity = tmp[:,7:]
        scale = torch.sigmoid(scale)

        # return opacity*10, scale, features, rotations
        return opacity*10, scale, rotations

    # def tv_loss(self, w):
    #     for level in range(self._feat.activate_level+1):
    #         factor = 1.0
    #         self._feat.k0s[level].total_variation_add_grad(w*((0.5)**(2-level)))
            

    # def calc_sparsity(self):

    #     plane = self._feat
    #     res = 0
    #     for level in range(self._feat.activate_level+1):
  
    #         factor = 1.0
            
    #         for data in [plane.k0s[level].xy_plane, plane.k0s[level].xz_plane, plane.k0s[level].yz_plane]:
    #             l1norm = torch.mean(torch.abs(data))
    #             res += l1norm * ((0.4)**(2-level)) * factor

    #     return res / ((self._feat.activate_level+1)*3)

class FeaturePlanes(nn.Module):
    def __init__(self, use_single_mlp, world_size, xyz_min, xyz_max, feat_dim = 24, mlp_width = [168], out_dim=[11], subplane_multiplier=1):
        super(FeaturePlanes, self).__init__()
        
        self.world_size, self.xyz_min, self.xyz_max = world_size, xyz_min, xyz_max
        # 激活层个数  因为三层是逐步训练的   slam时  三层一起训练  但应用不同的学习率
        self.activate_level = 1
        self.num_levels = 2
        self.level_factor = 0.5

        t_ws = torch.tensor(world_size)
        # k0s中存储3个三平面
        self.k0s =  torch.nn.ModuleList()

        plane_res = world_size
        xyz_len = xyz_max - xyz_min
        for grid_res in plane_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            self.k0s.append(PlaneGrid(feat_dim, grid_shape, xyz_min, xyz_max,config={'factor':1}))
            print('Grid Shape:', grid_shape)

        # 存储MLP网络
        # self.models = torch.nn.ModuleList()
        # plane_mlp = tcnn.Network(
        #     n_input_dims = feat_dim * self.num_levels,
        #     n_output_dims = 8,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 32,
        #         "n_hidden_layers": 2,
        #     },
        # )
        # self.tcmlp = plane_mlp.cuda()

        mlp_width = [mlp_width[0],mlp_width[0],mlp_width[0]] 
        out_dim = [out_dim[0],out_dim[0],out_dim[0]]

        # for i in range(self.num_levels):
        #     self.models.append(nn.Sequential(
        #                         nn.Linear(self.k0s[i].get_dim(), mlp_width[i]),
        #                         nn.ReLU(),
        #                         nn.Linear(mlp_width[i], mlp_width[i]),
        #                         nn.ReLU(),
        #                         nn.Linear(mlp_width[i], out_dim[i])
        #                         ))
        self.use_single_mlp = use_single_mlp
        self.single_mlp = nn.Sequential(nn.Linear(self.k0s[0].get_dim()*2, mlp_width[0]),
                                         nn.ReLU(),
                                            nn.Linear(mlp_width[0], mlp_width[0]),
                                            nn.ReLU(),
                                            nn.Linear(mlp_width[0], out_dim[0])
                                            )


    def forward(self, x, Q=0):
        # Pass the input through k0

        level_features = []
        # 从不同分辨率的三平面中解码feature
        for i in range(self.num_levels):
            feat = self.k0s[i](x , Q)
            level_features.append(feat)
        if self.use_single_mlp:
            # we concat coarse middle and fine level
            level_features = torch.cat(level_features, dim=-1)
            return self.single_mlp(level_features)
            # return self.tcmlp(level_features)
        # else:
        # # 将不同层级的feature分别送入独立的mlp
        #     res = []
        #     cnt =0
        #     for m,feat in zip(self.models,level_features):
        #         rr = m(feat)
        #         res.append(rr)
        #         cnt = cnt + 1
        #         if cnt>self.activate_level:
        #             break
            
        #     return sum(res)

# 三平面的类
class PlaneGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max,  config, residual_mode = False):
        super(PlaneGrid, self).__init__()
        if 'factor' in config:
            self.scale = config['factor']
        else:
            self.scale = 2
            
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.residual_mode = residual_mode
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        X = X*self.scale
        Y = Y*self.scale
        Z = Z*self.scale
        self.world_size = torch.tensor([X,Y,Z])
        R = self.channels
        Rxy = R
        # 定义其中每一个二维平面 维度是X*Y 每个维度上的特征数是R
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y ]).normal_(mean=0, std=0.01))
        self.xz_plane = nn.Parameter(torch.randn([1, R,  X, Z]).normal_(mean=0, std=0.01))
        self.yz_plane = nn.Parameter(torch.randn([1, R,  Y, Z]).normal_(mean=0, std=0.01))


        # self.quant = FakeQuantize()
        # self.quant.set_bits(12)



        print("Planes version activated !!!!!! ")

    
    # def quant_all(self):
    #     self.xy_plane.data = self.quant(self.xy_plane.data)
    #     self.xz_plane.data = self.quant(self.xz_plane.data)
    #     self.yz_plane.data = self.quant(self.yz_plane.data)




    def compute_planes_feat(self, ind_norm, Q ):
        # Interp feature (feat shape: [n_pts, n_comp])
        # 提取最后一个维度形如[y, x]
        xy_feat = F.grid_sample(self.xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        xz_feat = F.grid_sample(self.xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        yz_feat = F.grid_sample(self.yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
        # xy xz yz 三个平面的特征相加
        feat = xy_feat + xz_feat + yz_feat
        return feat       

    def forward(self, xyz, Q = 0, dir=None, center=None):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        # 确保xyz位于[-1, 1]
        # ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        # 给最后一个维度的元素后面加了一个0
        ind_norm = torch.cat([xyz, torch.zeros_like(xyz[...,[0]])], dim=-1)

       
        if self.channels > 1:
            out = self.compute_planes_feat(ind_norm, Q=Q)
            out = out.reshape(*shape,self.channels)
        else:
            raise Exception("no implement!!!!!!!!!!")
        return out

    # def scale_volume_grid(self, new_world_size):
    #     if self.channels == 0:
    #         return
    #     X, Y, Z = new_world_size
    #     X = X*self.scale
    #     Y=Y*self.scale
    #     Z = Z*self.scale

    #     self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
    #     self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
    #     self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))

    # def scale_volume_grid_value(self, new_world_size):
    #     if self.channels == 0:
    #         return
    #     X, Y, Z = new_world_size
    #     X = X*self.scale
    #     Y=Y*self.scale
    #     Z = Z*self.scale
    
    #     xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True), requires_grad=False)
    #     xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True), requires_grad=False)
    #     yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True), requires_grad=False)

    #     return xy_plane, xz_plane, yz_plane

    def get_dim(self):
        return self.channels

    def total_variation_add_grad(self, w):
        '''Add gradients by total variation loss in-place'''
        wx = wy= wz = w
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') 
        loss /= 6
        loss.backward()


    # def extra_repr(self):
    #     return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.channels //3}'


class Conctractor(nn.Module):
    def __init__(self, xyz_min, xyz_max, enable = True):
        super().__init__()
        self.enable = enable
        # if not self.enable:
        #     print('**Disable Contractor**')
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)

    def decontracte(self, xyz): 
        if not self.enable:
            raise Exception("Not implement")

        mask = torch.abs(xyz) > 1.0
        res = xyz.clone()
        signs = (res <0) & (torch.abs(res)>1.0)
        res[mask] = 1.0/(1.0- (torch.abs(res[mask])-1)) 
        res[signs] *= -1
        res = res * (self.xyz_max-self.xyz_min) /2 + (self.xyz_max+self.xyz_min) /2

        return res
    
    def contracte(self, xyz):
        indnorm = (xyz-self.xyz_min)*2.0 / (self.xyz_max-self.xyz_min) -1
        if self.enable:
            mask = torch.abs(indnorm)>1.0
            signs = (indnorm <0) & (torch.abs(indnorm)>1.0)
            indnorm[mask] = (1.0- 1.0/torch.abs(indnorm[mask])) +1.0
            indnorm[signs] *=-1
        return indnorm
    
def inference_gs(model, decoder, color_decoder, points):
    # 推理颜色
    xyz = contract_to_unisphere(points.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    dir_pp = (points - color_decoder['cam_center'].repeat(points.shape[0], 1))
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    shs = color_decoder['mlp_head'](torch.cat([color_decoder['recolor'](xyz), color_decoder['direction_encoding'](dir_pp)], dim=-1)).unsqueeze(1)
    # 推理opacity scales rotation
    feature = decoder(model(points.detach()))
    # split the feature
    opacity = feature[:,:1]
    opacity = opacity*10
    scales = feature[:,1:4]
    scales = torch.sigmoid(scales)
    scales = (scales-1)*5-2
    scales = scales.mean(dim=1 , keepdim=True)
    rotations = feature[:,4:8]
    
    return {
        'means3D': points,
        'shs': shs.float(),
        'unnorm_rotations': rotations,
        'logit_opacities': opacity,
        'log_scales': scales,
    }

@torch.no_grad()
def inference_gs_nograd(model, decoder, color_decoder, points):
    # 推理颜色
    xyz = contract_to_unisphere(points.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    dir_pp = (points - color_decoder['cam_center'].repeat(points.shape[0], 1))
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    shs = color_decoder['mlp_head'](torch.cat([color_decoder['recolor'](xyz), color_decoder['direction_encoding'](dir_pp)], dim=-1)).unsqueeze(1)
    # 推理opacity scales rotation
    feature = decoder(model(points.detach()))
    # split the feature
    opacity = feature[:,:1]
    opacity = opacity*10
    scales = feature[:,1:4]
    scales = torch.sigmoid(scales)
    scales = (scales-1)*5-2
    scales = scales.mean(dim=1 , keepdim=True)
    rotations = feature[:,4:8]
    
    return {
        'means3D': points,
        'shs': shs.float(),
        'unnorm_rotations': rotations,
        'logit_opacities': opacity,
        'log_scales': scales,
    }
def contract_to_unisphere(
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x