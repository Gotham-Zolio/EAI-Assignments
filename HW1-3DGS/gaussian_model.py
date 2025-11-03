#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid
from torch import nn
from plyfile import PlyData
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import strip_symmetric, build_rotmat_from_quat
from utils.sh_utils import eval_sh
            
class GaussianModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, active_sh_degree : int):
        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = 3  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        s= self.get_scaling # scaling vector
        q= self.get_rotation # rotation quaternion
        ## Begin code 2.1 ##
        # todo: build 3dgs covariance matrix. Equation (6) of 3dgs paper.
        R = build_rotmat_from_quat(q)
        S = torch.diag_embed(scaling_modifier * s)
        L = R @ S
        ## End code 2.1 ##
        
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def prune_points(self, th=0.0, mode=1, cameras=None):
        ## Begin code 3.2 ##
        opacity = self.get_opacity
        valid_points_mask = torch.ones((self.get_xyz.shape[0],), dtype=torch.bool, device=self._xyz.device)

        if mode == 1 or cameras is None:
            with torch.no_grad():
                valid_points_mask = (opacity.view(-1) > th)
                if valid_points_mask.sum().item() == 0:
                    return
                
        elif mode == 2:
            with torch.no_grad():
                importance = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device=self._xyz.device)
                means3D = self.get_xyz
                means2D = torch.zeros((self.get_xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device)
                cov3D_precomp = self.get_covariance(1)

                for cam in cameras:
                    tanfovx = math.tan(cam.FoVx * 0.5)
                    tanfovy = math.tan(cam.FoVy * 0.5)
                    raster_settings = GaussianRasterizationSettings(
                        image_height=int(cam.image_height),
                        image_width=int(cam.image_width),
                        tanfovx=tanfovx,
                        tanfovy=tanfovy,
                        bg=torch.tensor([0,0,0], dtype=torch.float32, device=self._xyz.device),
                        scale_modifier=1.0,
                        viewmatrix=cam.world_view_transform,
                        projmatrix=cam.full_proj_transform,
                        sh_degree=self.active_sh_degree,
                        campos=cam.camera_center,
                        prefiltered=False,
                        debug=False
                    )

                    # Precompute colors_per_point in this view (same as render path) so rasterizer
                    # receives exactly one of SHs or precomputed colors.
                    shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
                    dir_pp = (means3D - cam.camera_center.repeat(self.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
                    sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                    _, radii = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = None,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = None,
                        rotations = None,
                        cov3D_precomp = cov3D_precomp)

                    radii = radii.view(-1)
                    visible = (radii > 0).float()
                    area = (torch.pi * (radii ** 2)).float()
                    importance += (opacity.view(-1) * area * visible)

                cutoff = torch.quantile(importance, th)
                valid_points_mask = (importance > cutoff)
                if valid_points_mask.sum().item() == 0:
                    return

        ## End code 3.2 ##
        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        

