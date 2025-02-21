# lint as: python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from pydantic.dataclasses import dataclass
from flash_attn import flash_attn_func

@dataclass
class ModelConfig:
    max_seq_len: int
    num_layers: int = 32
    num_heads: int = 32 
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    vocab_size: int = 128256
    eps: float = 1e-05
    hidden_dim: int = 14336

    def calculate_token_flops(self):
        """Calculate training TFLOP per token"""
        head_dim = self.embedding_dim // self.num_heads

        # Compute FLOPs contributions (assuming implicit bsz=1)
        ffn1_flops = (
            2
            * self.max_seq_len
            * self.hidden_dim
            * self.embedding_dim
            * 2 # Factor of 2 for dual activations
        )
        ffn2_flops = 2 * self.max_seq_len * self.hidden_dim * self.embedding_dim
        total_ffn_flops = ffn1_flops + ffn2_flops

        qkv_flops = (
            2
            * self.max_seq_len
            * self.embedding_dim
            * (self.num_heads + 2 * self.num_kv_heads)
            * head_dim
        )
        attention_flops = 4 * self.max_seq_len**2 * self.num_heads * head_dim
        projection_flops = (
            2 * self.max_seq_len * self.embedding_dim * self.num_heads * head_dim
        )
        embedding_flops = 2 * self.max_seq_len * self.embedding_dim * self.vocab_size

        # multiply by 3 for both feed forward and backward passes
        learnable_weight_tflops = (
            ((total_ffn_flops + qkv_flops + projection_flops) * self.num_layers + embedding_flops) * 3
        )

        attention_tflops = attention_flops * self.num_layers * 3

        total_tflops = learnable_weight_tflops + attention_tflops
        self.flops_per_token =  total_tflops / self.max_seq_len

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[None, None, :, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RotaryAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_kv_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.kv_dim = embedding_dim * num_kv_heads // num_heads 
        self.in_proj = nn.Linear(embedding_dim, embedding_dim+2*self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self,input,position_encoding):
        qkv = self.in_proj(input)
        q,k,v = qkv.split([self.embedding_dim, self.kv_dim, self.kv_dim], -1)
        q = q.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [-1, self.head_dim]).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, position_encoding)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = flash_attn_func(q,k,v,dropout_p=0, causal=True)

        o = self.out_proj(o.reshape(input.shape))
        return o

class FeedForwardLayer(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
    
    def forward(self,input):
        hid = F.silu(self.gate_proj(input)) * self.up_proj(input)
        o = self.down_proj(hid)
        return o

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.eps = eps

    def forward(self, input):
        # use high precision, see https://github.com/foundation-model-stack/foundation-model-stack/blob/d55a9f2ade65ef4157cdfd928300874e2348e5d0/fms/modules/layernorm.py#L64
        input_float = input.float() 
        output = (input_float * torch.rsqrt(input_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(input) * self.weight
        return output

class FP8TransformerBlock(te.TransformerLayer):
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_kv_heads,eps):
        super().__init__(
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_gqa_groups=num_heads//num_kv_heads,
            fuse_qkv_params=True,
            attn_input_format='bshd',
            attention_dropout=0.0,
            normalization='RMSNorm',
            layernorm_epsilon=eps,
            ffn_hidden_size=hidden_dim,
            bias=False,
            activation='swiglu',
            hidden_dropout=0.0
        )
    
def generate_rotary_encoding(dim, max_seq_len):
    rope_base=500000.0
    assert dim % 2 == 0
    freqs = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))  # F = dim // 2
    t = torch.arange(max_seq_len, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)
    
class FP8Llama(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers,num_heads,num_kv_heads,max_seq_len,eps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        layers = []
        for i in range(num_layers):
            layers.append(FP8TransformerBlock(embedding_dim,hidden_dim,num_heads,num_kv_heads,eps))
        self.layers = nn.ModuleList(layers)
        self.norm_lm_head = te.LayerNormLinear(embedding_dim, vocab_size, bias=False,normalization='RMSNorm', eps=eps)

        position_encoding = te.attention.RotaryPositionEmbedding(embedding_dim//num_heads)(max_seq_len=max_seq_len)
        self.register_buffer('position_encoding', position_encoding.to(torch.bfloat16))

    def forward(self, idxs, is_first_microbatch):
        x = self.embedding(idxs)
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=self.position_encoding, is_first_microbatch=is_first_microbatch)
        logits = self.norm_lm_head(x)
        return logits
