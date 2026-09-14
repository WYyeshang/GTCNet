import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class MultiHeadAttention(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, S, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_weights: bool = False,
    ):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = self._merge_heads(torch.matmul(attn_weights, V))
        out = self.W_o(out)

        if return_weights:
            return out, attn_weights.mean(dim=1)
        return out




class LowRankAttention(nn.Module):
   

    def __init__(self, d_model: int, n_heads: int, max_seq: int, rank: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.rank = rank

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.E_k = nn.Parameter(torch.empty(n_heads, max_seq, rank))
        self.E_v = nn.Parameter(torch.empty(n_heads, max_seq, rank))
        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, S, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, S_q = query.shape[0], query.shape[1]
        S_k = key.shape[1]

        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        E_k_batch = self.E_k[:, :S_k, :]
        E_v_batch = self.E_v[:, :S_k, :]

        # E: [H, S, R], K/V: [B, H, S, D] -> proj: [B, H, R, D]
        K_proj = torch.einsum("hsr,bhsd->bhrd", E_k_batch, K)
        V_proj = torch.einsum("hsr,bhsd->bhrd", E_v_batch, V)

        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return self.W_o(self._merge_heads(torch.matmul(attn_weights, V_proj)))





class GroupedChannelAttention(nn.Module):
   

    def __init__(self, d_model: int, n_heads: int, n_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        BP, C, D = x.shape
        G = min(self.n_groups, C)
        group_size = math.ceil(C / G)

        pad_len = 0
        if C % G != 0:
            pad_len = G * group_size - C
            x = F.pad(x, (0, 0, 0, pad_len))

        C_padded = C + pad_len
        # [BP, G, group_size, D] -> [BP*G, group_size, D]
        x_r = x.reshape(BP, G, group_size, D).reshape(BP * G, group_size, D)
        out = self.attention(x_r, x_r, x_r)
        out = out.view(BP, G, group_size, D).reshape(BP, C_padded, D)

        if pad_len > 0:
            out = out[:, :C, :]
        return out


def channel_shuffle(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    
    B, C, P, D = x.shape
    G = n_groups

    if C % G != 0:
        pg_size = (C + G - 1) // G
        pad_len = G * pg_size - C
        x_p = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        x_p = x_p.view(B, G, pg_size, P, D).transpose(1, 2).contiguous()
        x_p = x_p.view(B, G * pg_size, P, D)
        return x_p[:, :C, :, :]
    else:
        gs = C // G
        return x.view(B, G, gs, P, D).transpose(1, 2).contiguous().view(B, C, P, D)





class ChannelAwareTemporalEncoder(nn.Module):
  
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.temporal_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.temporal_ln = nn.LayerNorm(d_model)

        self.W_stats = nn.Linear(d_model * 2, d_model)
        self.cross_channel_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.W_s = nn.Linear(d_model, d_model // 4)
        self.v_s = nn.Linear(d_model // 4, 1)

    def forward(self, h: torch.Tensor, h_input: torch.Tensor) -> torch.Tensor:
        
        B, C, P, D = h.shape

        
        h_r = h.reshape(B * C, P, D)
        h_attn = self.temporal_attn(h_r, h_r, h_r)
        h_temp = self.temporal_ln(h_r + h_attn).view(B, C, P, D)

        
        h_mean = h_temp.mean(dim=2)  
        h_var = h_temp.var(dim=2, unbiased=False)
        h_stats = self.W_stats(torch.cat([h_mean, h_var], dim=-1))  

       
        H_resh = h_temp.permute(0, 2, 1, 3).contiguous().view(B * P, C, D)
        h_stats_exp = h_stats.unsqueeze(1).expand(B, P, C, D).contiguous().view(B * P, C, D)

        A = self.cross_channel_attn(h_stats_exp, H_resh, H_resh)  
        A_pool = A.view(B, P, C, D).mean(dim=1)  

        
        fused = A_pool + h_stats  
        logits = self.v_s(F.gelu(self.W_s(fused))).squeeze(-1) 
        s = F.softmax(logits / self.temperature, dim=-1) 
        s = s.unsqueeze(-1).unsqueeze(-1) 

       
        return h_input + s * (h_temp - h_input)


class ChannelIndependentPath(nn.Module):
    
    def __init__(self, d_model: int, mlp_ratio: int = 4, conv_kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_model * mlp_ratio

        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel // 2, groups=d_model
        )
        self.conv_ln = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.mlp_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        B, C, P, D = x.shape
        h = x.reshape(B * C, P, D)

        h_c = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv_ln(h + h_c)

       
        h = self.mlp_ln(h + self.mlp(h))

        return h.reshape(B, C, P, D)



class ChannelMixingPath(nn.Module):
   
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_channels: int = 862,
        linformer_rank: int = 64,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linformer_rank = linformer_rank
        self.n_groups = n_groups

        self.standard_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.lowrank_attn = LowRankAttention(d_model, n_heads, max_channels, linformer_rank, dropout)
        self.grouped_attn = GroupedChannelAttention(d_model, n_heads, n_groups, dropout)

        self.attn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        B, C, P, D = x.shape

      
        h = x.permute(0, 2, 1, 3).contiguous().view(B * P, C, D)  
           
        h_a = self.grouped_attn(h)

        h = self.attn_ln(h + h_a)
        h = self.ffn_ln(h + self.ffn(h))

        return h.view(B, P, C, D).permute(0, 2, 1, 3).contiguous()


class TemporalGatedChannelMixer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_channels: int,
        mlp_ratio: int = 4,
        linformer_rank: int = 64,
        n_groups: int = 8,
        dropout: float = 0.1,
       
    ):
        super().__init__()
        
        
        self.W_pool = nn.Linear(d_model * 2, d_model)
        self.context_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.context_ln = nn.LayerNorm(d_model)

        
        self.mlp_ch = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.mlp_global = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.ci_path = ChannelIndependentPath(d_model, mlp_ratio, 3, dropout)
        self.cm_path = ChannelMixingPath(d_model, n_heads, max_channels, linformer_rank, n_groups, dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
       
        B, C, P, D = h.shape
       
        mu = h.mean(dim=1)  
        mx = h.max(dim=1).values
        C0 = self.W_pool(torch.cat([mu, mx], dim=-1))
        C_ctx = self.context_ln(C0 + self.context_attn(C0, C0, C0))

        C_exp = C_ctx.unsqueeze(1).expand(B, C, P, D)
        g_ch = self.mlp_ch(torch.cat([C_exp, h], dim=-1))  
        g_gl = self.mlp_global(C_ctx).unsqueeze(1)         
        gate = torch.sigmoid(g_ch + g_gl)                   

        h_ci = self.ci_path(h)
        h_cm = self.cm_path(h)

        h_out = gate * h_cm + (1.0 - gate) * h_ci
        return h_out, gate

class GTCBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_channels: int,
        temperature: float = 0.5,
        mlp_ratio: int = 4,
        linformer_rank: int = 64,
        n_groups: int = 8,
        dropout: float = 0.1,
        do_channel_shuffle: bool = False,
        
    ):
        super().__init__()
        self.do_channel_shuffle = do_channel_shuffle
        self.n_groups = n_groups
        

        self.cate = ChannelAwareTemporalEncoder(d_model, n_heads, temperature, dropout)
        self.tgcm = TemporalGatedChannelMixer(
            d_model=d_model,
            n_heads=n_heads,
            max_channels=n_channels,
            mlp_ratio=mlp_ratio,
            linformer_rank=linformer_rank,
            n_groups=n_groups,
            dropout=dropout,
            
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                      
        h_cate = self.cate(h, h)

        if self.do_channel_shuffle:
            h_cate = channel_shuffle(h_cate, self.n_groups)

        h_out, gate = self.tgcm(h_cate)
        return h_out, gate

class CrossScaleAttentionFusion(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, scale_features: list[torch.Tensor]) -> list[torch.Tensor]:
       
        B = scale_features[0].shape[0]
        C = scale_features[0].shape[1]
        D = scale_features[0].shape[3]
        S = len(scale_features)

        globals_list = [h_s.mean(dim=2) for h_s in scale_features]
 
        g_stack = torch.stack(globals_list, dim=1).permute(0, 2, 1, 3).contiguous().view(B * C, S, D)
        g_out = self.ln(g_stack + self.attn(g_stack, g_stack, g_stack))
        g_out = g_out.view(B, C, S, D).permute(0, 2, 1, 3)  

        fused = []
        for s_idx, h_s in enumerate(scale_features):
            fused.append(h_s + g_out[:, s_idx, :, :].unsqueeze(2))

        return fused
class AdaptiveScaleFusion(nn.Module):

    def __init__(self, d_model: int, n_scales: int):
        super().__init__()
        self.W_score = nn.Linear(d_model, 1)

    def forward(
        self, scale_features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        globals_list = [h_s.mean(dim=2) for h_s in scale_features]
        stacked = torch.stack(globals_list, dim=2)  
        logits = self.W_score(stacked).squeeze(-1)
        weights = F.softmax(logits, dim=-1)  

        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return fused, weights

class GTCNetLoss(nn.Module):

    def __init__(
        self,
        lambda_gate: float = 1e-4,
        lambda_scale: float = 1e-5,
        use_ms: bool = False,
    ):
        super().__init__()
        self.lambda_gate = lambda_gate
        self.lambda_scale = lambda_scale
        self.use_ms = use_ms
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gates: Optional[List] = None,
        scale_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}
        loss = self.mse(pred, target)
        loss_dict["mse"] = loss.item()

        if gates is not None and self.lambda_gate > 0:
            g_loss = self._gate_reg(gates)
            loss = loss + self.lambda_gate * g_loss
            loss_dict["gate"] = g_loss.item()

        if self.use_ms and scale_weights is not None and self.lambda_scale > 0:
            s_loss = self._scale_reg(scale_weights)
            loss = loss + self.lambda_scale * s_loss
            loss_dict["scale"] = s_loss.item()

        loss_dict["total"] = loss.item()
        return loss, loss_dict

    @staticmethod
    def _gate_reg(gates: Optional[List]) -> torch.Tensor:
        flat = []
        for g in gates:
            if isinstance(g, list):
                flat.extend(g)
            else:
                flat.append(g)
        if not flat:
            return torch.tensor(0.0)

        var_loss, ent_loss, n = 0.0, 0.0, 0
        for g in flat:
            g = g.squeeze(-1)  # [B, C, P]
            var_loss += -g.var(dim=-1).mean()
            eps = 1e-7
            ent_loss += -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps)).mean()
            n += 1
        return (var_loss / max(n, 1)) + (ent_loss / max(n, 1))

    @staticmethod
    def _scale_reg(w: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        return (w * torch.log(w + eps)).sum(dim=-1).mean()
