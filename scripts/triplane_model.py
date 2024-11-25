import torch
from torch import nn
import torch.nn.functional as F
class GaussianLearner(nn.Module):
    def __init__(self, model_params, xyz_min = [-2, -2, -2], xyz_max=[2, 2, 2] ):
        super(GaussianLearner, self).__init__()

        self.Q0 = 0.03
        xyz_min = [model_params['xmin'], model_params['ymin'], model_params['zmin']]
        xyz_max = [model_params['xmax'], model_params['ymax'], model_params['zmax']]
        self.xyz_min = torch.tensor(xyz_min).cuda()
        self.xyz_max = torch.tensor(xyz_max).cuda()

        self.world_size = [model_params['plane_size']]*3
        self.max_step = 6
        self.current_step = 0

        self._feat = FeaturePlanes(world_size=self.world_size, xyz_min = self.xyz_min, xyz_max= self.xyz_max,
                                    feat_dim = model_params['num_channels'], mlp_width = [model_params['mlp_dim']], out_dim=[35], subplane_multiplier=model_params['subplane_multiplier'] )  # 27,4,3,1

        self.register_buffer('opacity_scale', torch.tensor(10))
        self.opacity_scale = self.opacity_scale.cuda()

        # self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()


    def activate_plane_level(self):
        self._feat.activate_level +=1
        print('******* Plane Level to:', self._feat.activate_level)


    def inference(self, xyz):
        inputs = xyz.cuda().detach()
        
        tmp  = self._feat(inputs, self.Q0)
        # features = tmp[:,:3]
        # rotations = tmp[:,3:3+4]
        # scale = tmp[:,7:7+3]
        # opacity = tmp[:,10:]
        # scale = torch.sigmoid(scale)
        features = tmp[:,:27]
        rotations = tmp[:,27:27+4]
        scale = tmp[:,31:31+3]
        opacity = tmp[:,34:]
        scale = torch.sigmoid(scale)

        return opacity*10, scale, features, rotations

    def tv_loss(self, w):
        for level in range(self._feat.activate_level+1):
            factor = 1.0
            self._feat.k0s[level].total_variation_add_grad(w*((0.5)**(2-level)))
            

    def calc_sparsity(self):

        plane = self._feat
        res = 0
        for level in range(self._feat.activate_level+1):
  
            factor = 1.0
            
            for data in [plane.k0s[level].xy_plane, plane.k0s[level].xz_plane, plane.k0s[level].yz_plane]:
                l1norm = torch.mean(torch.abs(data))
                res += l1norm * ((0.4)**(2-level)) * factor

        return res / ((self._feat.activate_level+1)*3)

class FeaturePlanes(nn.Module):
    def __init__(self, world_size, xyz_min, xyz_max, feat_dim = 24, mlp_width = [168], out_dim=[11], subplane_multiplier=1):
        super(FeaturePlanes, self).__init__()
        
        self.world_size, self.xyz_min, self.xyz_max = world_size, xyz_min, xyz_max
        # 激活层个数  因为三层是逐步训练的   slam时  三层一起训练  但应用不同的学习率
        self.activate_level = 2
        self.num_levels = 3
        self.level_factor = 0.5

        t_ws = torch.tensor(world_size)
        # k0s中存储3个三平面
        self.k0s =  torch.nn.ModuleList()

        for i in range(self.num_levels):
            cur_ws = (t_ws*self.level_factor**(self.num_levels-i-1)).cpu().int().numpy().tolist()
            self.k0s.append(PlaneGrid(feat_dim, cur_ws, xyz_min, xyz_max,config={'factor':1}))
            print('Create Planes @ ', cur_ws)

        # 存储MLP网络
        self.models = torch.nn.ModuleList()

        mlp_width = [mlp_width[0],mlp_width[0],mlp_width[0]] 
        out_dim = [out_dim[0],out_dim[0],out_dim[0]]

        for i in range(self.num_levels):
            self.models.append(nn.Sequential(
                                nn.Linear(self.k0s[i].get_dim(), mlp_width[i]),
                                nn.ReLU(),
                                nn.Linear(mlp_width[i], mlp_width[i]),
                                nn.ReLU(),
                                nn.Linear(mlp_width[i], out_dim[i])
                                ))


    def forward(self, x, Q=0):
        # Pass the input through k0

        level_features = []

        for i in range(self.activate_level + 1):
            feat = self.k0s[i](x , Q)
            level_features.append(feat)
        # 将不同层级的feature分别送入独立的mlp
        res = []
        cnt =0
        for m,feat in zip(self.models,level_features):
            rr = m(feat)
            res.append(rr)
            cnt = cnt + 1
            if cnt>self.activate_level:
                break
        
        return sum(res)

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
        R = self.channels //3
        Rxy = R
        # 定义其中每一个二维平面 维度是X*Y 每个维度上的特征数是R
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y ]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R,  X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R,  Y, Z]) * 0.1)


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

        # Aggregate components
        # Q不等于0时会加入噪声
        if Q == 0:
            feat = torch.cat([
                xy_feat ,
                xz_feat ,
                yz_feat
            ], dim=-1)
        else:
            feat = torch.cat([
                xy_feat + torch.empty_like(xy_feat).uniform_(-0.5, 0.5) * Q,
                xz_feat + torch.empty_like(xz_feat).uniform_(-0.5, 0.5) * Q,
                yz_feat + torch.empty_like(yz_feat).uniform_(-0.5, 0.5) * Q 
            ], dim=-1)


        return feat       

    def forward(self, xyz, Q = 0, dir=None, center=None):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        # 确保xyz位于[-1, 1]
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        # 给最后一个维度的元素后面加了一个0
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[...,[0]])], dim=-1)

       
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
        if not self.enable:
            print('**Disable Contractor**')
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
    
def inference_gs(model, points):
    if model['enable_net']:
        opacity_net, scales_net, rgb_net, rotations_net = model['tri_plane'].inference(model['contractor'].contracte(points))
        scales_net = (scales_net-1)*5-2
        scales_net = scales_net.mean(dim=1 , keepdim=True)
        # rgb_net = F.normalize(rgb_net, dim=1)
        rgb_net = rgb_net.view(rgb_net.size(0), (model['max_sh_degree'] + 1) ** 2, 3)
        feature_dc = rgb_net[:,0:1,:]
        feature_rest = rgb_net[:,1:,:]
        params_net = {
        'means3D': points,
        'shs': torch.cat([feature_dc, feature_rest], dim=1),
        'unnorm_rotations': rotations_net,
        'logit_opacities': opacity_net,
        'log_scales': scales_net,
        }
        return params_net
    return {}

@torch.no_grad()
def inference_gs_nograd(model, points):
    if model['enable_net']:
        opacity_net, scales_net, rgb_net, rotations_net = model['tri_plane'].inference(model['contractor'].contracte(points.detach()))
        scales_net = (scales_net-1)*5-2
        scales_net = scales_net.mean(dim=1 , keepdim=True)
        rgb_net = F.normalize(rgb_net, dim=1)
        params_net = {
        'means3D': points.detach(),
        'rgb_colors': rgb_net,
        'unnorm_rotations': rotations_net,
        'logit_opacities': opacity_net,
        'log_scales': scales_net,
        }
        return params_net
    return {}