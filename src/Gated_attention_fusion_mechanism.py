import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultiHeadCrossAttention(nn.Module):
    '''
     双向门控多头交叉注意力机制
    '''
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Gate networks
        self.gate_q = nn.Linear(dim, dim)
        self.gate_c = nn.Linear(dim, dim)

        # Final projection after multi-head concat
        self.out_proj = nn.Linear(dim, dim)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward_one_way(self, q_input, kv_input):
        B, N, _ = q_input.size()

        # Linear projections and reshape to (B, heads, N, head_dim)
        Q = self.q_proj(q_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Concat heads
        context = context.transpose(1, 2).contiguous().view(B, N, self.dim)

        # Gate
        gate = torch.sigmoid(self.gate_q(q_input) + self.gate_c(context))  # (B, N, dim)
        gated_output = gate * context + (1 - gate) * q_input  # Gated fusion

        # Residual + Norm + Dropout
        fused = self.dropout(self.out_proj(gated_output))
        output = self.norm(q_input + fused)

        return output

    def forward(self, h1, h2):
        # h1: EGNN output, h2: GINE output
        out1 = self.forward_one_way(h1, h2)  # EGNN attends to GINE
        out2 = self.forward_one_way(h2, h1)  # GINE attends to EGNN
        return out1, out2

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(GatedMultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)  # 第一个层归一化
        self.dropout1 = nn.Dropout(dropout)  # 注意力后的dropout

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 简单的前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),  # 扩大维度
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # FFN内部的dropout
            nn.Linear(hidden_size * 4, hidden_size)  # 还原维度
        )
        self.norm2 = nn.LayerNorm(hidden_size)  # 第二个层归一化
        self.dropout2 = nn.Dropout(dropout)  # FFN后的dropout

    def forward(self, x):  # [B, L, H]
        # 注意力层
        attn_output, _ = self.multihead_attn(x, x, x)  # [B, L, H]
        attn_output = self.dropout1(attn_output)

        # 残差连接 + 层归一化 (第一个)
        x = self.norm1(x + attn_output)  # x现在是经过注意力处理和归一化的结果

        # 门控机制
        gate = self.gate(x)  # 门控基于经过注意力处理的x
        gated_output = gate * x  # 将门控应用到注意力输出（这里是x，因为它已经包含了注意力输出和残差）
        # 或者 gate * attn_output 如果你希望门控仅作用于attn_output
        # 我这里选择 gate * x 是因为x现在是attn_output的残差连接结果，更符合常规Transformer block的flow

        # 前馈网络
        ffn_output = self.ffn(gated_output)  # FFN应用于门控后的输出
        ffn_output = self.dropout2(ffn_output)

        # 残差连接 + 层归一化 (第二个)
        output = self.norm2(gated_output + ffn_output)  # FFN后的残差连接和归一化

        # 根据需求决定是否进行平均池化
        # 如果是处理序列的每个元素，则直接返回 output
        # 如果是为整个序列生成一个固定大小的表示，且序列长度可能大于1，则使用 mean(dim=1)
        # 如果序列长度总是1，那么 mean(dim=1) 可以简化为 squeeze(dim=1)
        # return output.mean(dim=1)  # 示例中保持原有的mean操作
        return output

# -----------------------------
# AttentionModel Wrapper
# -----------------------------
class GatedAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, device='cuda', num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.attn = GatedMultiHeadAttention(hidden_size, num_heads=num_heads)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N, 1, D]
        x = self.embedding(x)   # [N, 1, H]
        out = self.attn(x)      # [N, H]
        return self.output_proj(out)

class FinalAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, device='cuda', num_heads=4, num_layers=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        num_intermediate_layers = max(0, num_layers - 1)

        attention_modules = [GatedMultiHeadCrossAttention(input_size, num_heads=num_heads) for _ in range(num_intermediate_layers)]

        if num_intermediate_layers > 0:
            attention_modules.append(GatedAttentionModel(input_size*4, 8 * hidden_size, num_heads=num_heads))

        self.attention_block = nn.ModuleList(attention_modules)

    def forward(self, x1, x2):
        shortcut1, shortcut2 = x1, x2
        for at_layer in self.attention_block[:-1]:
            x1, x2 = at_layer(x1,x2)

        x1 = torch.cat((x1, shortcut1), dim=2)
        x2 = torch.cat((x2, shortcut2), dim=2)
        x = torch.cat((x1, x2), dim=2)
        shortcut = x
        x = self.attention_block[-1](x)
        return x + shortcut