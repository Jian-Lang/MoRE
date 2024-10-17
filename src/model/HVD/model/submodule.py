import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = attn_weights.squeeze(-1)  # (batch_size, seq_len)

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len)

        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, input_dim)

        return pooled
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.d_model = d_model
        
        self.init_weights()
        
    def init_weights(self):
        position = torch.arange(0, self.positional_encoding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.positional_encoding.size(1), 2) * 
                             -(math.log(10000.0) / self.positional_encoding.size(1)))
        
        pe = torch.zeros_like(self.positional_encoding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding.data.copy_(pe)

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 4:
            x = x.view(-1, original_shape[2], self.d_model)
            
        x = x + self.positional_encoding[:x.size(1), :]
        
        if len(original_shape) == 4:
                x = x.view(original_shape)
                
        return x
        
def check_shape(x, retreival):
    if not retreival:
        # B, L, D
        if len(x.shape) == 3:
            x = x
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f'Invalid shape: {x.shape}')
    else:
        # B, N, L, D
        if len(x.shape) == 4:
            x = x
        elif len(x.shape) == 3:
            x = x.unsqueeze(2)
        else:
            raise ValueError(f'Invalid shape: {x.shape}')
    return x