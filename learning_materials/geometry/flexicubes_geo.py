import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from render import mesh
from render import render
from render import regularizer
from render import util

import render.optixutils as ou

from geometry.flexicubes import FlexiCubes

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff


###############################################################################
#  Geometry interface
###############################################################################

class FlexiCubesGeometry:
    def __init__(self, grid_res, scale, FLAGS):
        super(FlexiCubesGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.flexicubes    = FlexiCubes()
        verts, indices     = self.flexicubes.construct_voxel_grid(grid_res) 

        n_cubes = indices.shape[0]
        per_cube_weights = torch.ones((n_cubes, 21),dtype=torch.float,device='cuda')

        self.verts    = verts * scale
        self.indices  = indices
        
        print("FlexiCubes grid min/max", torch.min(self.verts).item(), torch.max(self.verts).item())
        self.generate_edges()

        # Random init
        sdf = torch.rand_like(self.verts[:,0]) - 0.1

        #self.mlpsdf = MLPSDF()

        self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.per_cube_weights = torch.nn.Parameter(torch.ones_like(per_cube_weights, requires_grad=True))
        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)

        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

    def generate_edges(self):
        with torch.no_grad():
            edges = self.flexicubes.cube_edges
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            self.max_displacement = util.length(self.verts[self.all_edges[:, 0]] - self.verts[self.all_edges[:, 1]]).mean() / 4

    def parameters(self):
        
        return [self.sdf, self.deform, self.per_cube_weights]  
    def getOptimizer(self, lr_pos):
        return torch.optim.Adam(self.parameters(), lr=lr_pos)
    
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    @torch.no_grad()
    def map_uv2(self, faces):
        uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float, device='cuda')
        uv_idx = torch.tensor([0,1,2], dtype=torch.long, device='cuda').repeat(faces.shape[0],1)
        return uvs, uv_idx

    @torch.no_grad()
    def map_uv(self, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='floor')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, N, rounding_mode='floor'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    def getMesh(self, material, _training=False):

        # Run FlexiCubes to get a base mesh
        v_deformed = self.verts + self.max_displacement * torch.tanh(self.deform)
        #out = self.mlpsdf.sample(self.verts)
        #sdf = out[..., 0]
        #print(sdf.shape)
        #self.sdf = sdf
        #print(self.sdf.min())
        #deform = out[..., 1:4]
        #self.deform  = deform
        #v_deformed = self.verts + self.max_displacement * torch.tanh(deform)
        #v_deformed = self.verts 
        verts, faces, reg_loss = self.flexicubes(v_deformed, self.sdf, self.indices, self.grid_res, 
                            self.per_cube_weights[:,:12], self.per_cube_weights[:,12:20], self.per_cube_weights[:,20],
                            training=_training)

        self.flexi_reg_loss = reg_loss.mean()

        face_gidx = torch.arange(faces.shape[0], dtype=torch.long, device="cuda")
        uvs, uv_idx = self.map_uv(face_gidx, faces.shape[0])

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        imesh = mesh.get_kdks(imesh)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)


        return imesh

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, FLAGS):

        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        opt_mesh = self.getMesh(opt_material)
        shadow_ramp = min(iteration / 1750, 1.0)

        buffers = render.render_mesh(FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], target['light'] if lgt is None else lgt, target['resolution'],
                                    spp=target['spp'], num_layers=FLAGS.layers, msaa=True, background=target['background'], optix_ctx=self.optix_ctx,  shadow_scale=shadow_ramp)

        img_loss  = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01

        reg_loss += regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, FLAGS.lambda_diffuse, FLAGS.lambda_specular)
        
        # Material smoothness regularizer
        reg_loss += regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)
        reg_loss += regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)

        reg_loss += self.flexi_reg_loss* 0.25
        assert 'perturbed_nrm' not in buffers

        return img_loss, reg_loss
    

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


class MLPSDF(torch.nn.Module):
    def __init__(self, channels = 4, internal_dims = 32, hidden = 2, min_max = None):
        super(MLPSDF, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.min_max = min_max

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }

        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        #self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    # Sample texture at a given location
    def sample(self, texc):
        _texc = texc.reshape(-1, 3)
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)
        self.min_max = [torch.tensor([-1, -1, -1, -1], dtype=torch.float32, device='cuda', requires_grad=False),torch.tensor([1, 1, 1, 1], dtype=torch.float32, device='cuda', requires_grad=False)]
        #out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]
        return out
    
    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()
"""
def hook_function(module, grad_i, grad_o):
    if grad_i[0] is not None:
        gradient_scaling = 10.0  # 你自己定义的梯度缩放因子
        return (grad_i[0] / gradient_scaling, )
    else:
        return (None, )
"""

