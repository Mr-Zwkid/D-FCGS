import torch
import tinycudann as tcnn

ntc_conf = {
    "loss": {
        "otype": "RelativeL2Luminance"
    },
    "optimizer": {
        "otype": "Adam",
        "learning_rate": 1e-3,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1e-15,
        "l2_reg": 1e-6
    },
    "encoding": {
                "otype": "HashGrid",
                "n_dims_to_encode": 3,
                "per_level_scale": 2.0,
                "log2_hashmap_size": 15,
                "base_resolution": 16,
                "n_levels": 16,
                "n_features_per_level": 4
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    },
    "others": {
        "otype": "EMA",
        "decay": 0.99,
        "nested": {
            "otype": "Adam",
            "learning_rate": 1e-2,
            "beta1": 0.9,
            "beta2": 0.99,
            "epsilon": 1e-15,
            "l2_reg": 1e-6
        }
    }
}

class NeuralTransformationCache(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max):
        super(NeuralTransformationCache, self).__init__()
        self.model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        
    def dump(self, path):
        torch.save(self.state_dict(),path)
        
    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz
        
    def forward(self, xyz:torch.Tensor):
        contracted_xyz=self.get_contracted_xyz(xyz)
        
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        
        ntc_inputs=torch.cat([contracted_xyz[mask]],dim=-1)
        resi=self.model(ntc_inputs)
        
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        # masked_d_opacity=resi[:,7:None]
        
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.half, device="cuda")
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.half, device="cuda")
        d_rot[:, 0] = 1.0
        # d_opacity = self._origin_d_opacity.clone()

        d_xyz[mask] = masked_d_xyz
        d_rot[mask] = masked_d_rot
        
        return mask, d_xyz, d_rot
        