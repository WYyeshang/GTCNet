import torch
import torch.nn as nn

from layers.gtc_layers import (
    GTCBlock,
    CrossScaleAttentionFusion,
    AdaptiveScaleFusion,
    GTCNetLoss,
)
from layers.revin import RevIN

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_channels = configs.enc_in

        self.patch_len = getattr(configs, "patch_len", 16)
        self.stride = getattr(configs, "stride", 8)
        self.padding_patch = getattr(configs, "padding_patch", "end")

        self.d_model = getattr(configs, "d_model", 256)
        self.n_heads = getattr(configs, "n_heads", 8)
        self.n_layers = getattr(configs, "n_layers", 2)
        self.temperature = getattr(configs, "temperature", 0.5)
        self.mlp_ratio = getattr(configs, "mlp_ratio", 4)
        self.linformer_rank = getattr(configs, "linformer_rank", 64)
        self.n_groups = getattr(configs, "n_groups", 8)
        self.dropout = getattr(configs, "dropout", 0.1)

        self.revin = bool(getattr(configs, "revin", 1))
        if self.revin:
            self.revin_layer = RevIN(self.n_channels, affine=True, subtract_last=False)

        self._compute_patch_counts()

        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, self.n_patches, self.d_model) * 0.02
            )

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)

        self.patch_dropout = nn.Dropout(self.dropout)

        self._build_ss_blocks()

       
        self.final_ln = nn.LayerNorm(self.d_model)
        
        self._build_ss_head()

        self._init_weights()

    def _compute_patch_counts(self):
        
        if self.use_ms:
            self.n_patches_per_scale = []
            for pl, st in zip(self.ms_patch_lens, self.ms_strides):
                n = (self.seq_len - pl) // st + 1
                if self.padding_patch == "end":
                    n += 1
                self.n_patches_per_scale.append(n)
            self.n_patches = self.n_patches_per_scale[0]
        else:
            self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1
            if self.padding_patch == "end":
                self.n_patches += 1

    def _build_ss_blocks(self):
        """Build single-scale GTC blocks."""
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            do_shuffle = (self.n_channels > 200) and (layer_idx % 2 == 1)

            print(
            f">>> Build GTCBlock {layer_idx}: "
            f"Channel Shuffle = {'ENABLED' if do_shuffle else 'DISABLED'} "
            f"(C={self.n_channels})"
            )
            self.blocks.append(
                GTCBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_channels=self.n_channels,
                    temperature=self.temperature,
                    mlp_ratio=self.mlp_ratio,
                    linformer_rank=self.linformer_rank,
                    n_groups=self.n_groups,
                    dropout=self.dropout,
                    do_channel_shuffle=do_shuffle,
                    
                )
            )

    def _build_ss_head(self):
        """Single-scale prediction head."""
        flat_dim = self.n_patches * self.d_model
        self.pred_head = nn.Sequential(
            nn.Linear(flat_dim, flat_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(flat_dim // 2, self.pred_len),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def _apply_patching(self, x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
        
        if self.padding_patch == "end":
            x = nn.functional.pad(x, (0, stride), mode="replicate")
        return x.unfold(dimension=-1, size=patch_len, step=stride)

    
    def forward(self, x, return_aux: bool = False):
        
        B = x.shape[0]
        
        if self.revin:
            x_norm = self.revin_layer(x, "norm")
        else:
            x_norm = x
        x_norm = x_norm.permute(0, 2, 1)  # [B, C, L]
        
        h, gates_all = self._forward_ss_blocks(x_norm)

        y_raw = self._forward_ss_head(h, B)
        scale_weights = None

        if self.revin:
            y = self.revin_layer(y_raw, "denorm")
        else:
            y = y_raw

        if return_aux:
            aux = {"gates": gates_all}
            if scale_weights is not None:
                aux["scale_weights"] = scale_weights
            return y, aux
        return y

    def _forward_ss_blocks(self, x_norm: torch.Tensor):
        """Single-scale patching + GTC blocks."""
        B, C, L = x_norm.shape

        # Patching
        h = self._apply_patching(x_norm, self.patch_len, self.stride)  # [B, C, P, patch_len]
        P = h.shape[2]
        h = self.patch_proj(h)  # [B, C, P, D]
        h = h + self.pos_embed[:, :, :P, :]
        h = self.patch_dropout(h)

        # GTC Blocks
        gates_all = []
        for block in self.blocks:
            h, gate = block(h)
            gates_all.append(gate)

        return h, gates_all

    def _forward_ss_head(self, h: torch.Tensor, B: int):
        """Single-scale prediction head."""
      
        h = self.final_ln(h.permute(0, 2, 1, 3).contiguous())  
        h = h.permute(0, 2, 1, 3).contiguous()  
        h_flat = h.reshape(B, self.n_channels, -1)  
        y_raw = self.pred_head(h_flat)  
        return y_raw.permute(0, 2, 1)  

def create_loss(configs):
    """Create GTCNetLoss from configs."""
    return GTCNetLoss(
        lambda_gate=getattr(configs, "lambda_gate", 1e-4),
        lambda_scale=getattr(configs, "lambda_scale", 1e-5),
        use_ms=bool(getattr(configs, "use_ms", 0)),
    )
