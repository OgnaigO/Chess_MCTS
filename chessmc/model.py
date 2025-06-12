import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.nn import GroupNorm

NUM_PLANES = 18
class ImprovedChessDataset(Dataset):
    def __init__(self, data_path, start_idx=0, end_idx=None):
        data = np.load(data_path)
        self.inputs = data['arr_0'][start_idx:end_idx]
        self.outputs = data['arr_1'][start_idx:end_idx]
        print(f'Data loaded: {self.inputs.shape}, {self.outputs.shape}')

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        # Reshape từ 768 về 8x8x12 (height x width x channels)
        board_tensor = self.inputs[index].reshape(12, 8, 8)
        return board_tensor, self.outputs[index]


class ResidualBlock(nn.Module):
    """Residual block với BatchNorm và skip connections"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AttentionBlock(nn.Module):
    """Self-attention để focus vào các vùng quan trọng"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Tạo query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # Attention weights
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class ImprovedChessModel(nn.Module):
    """
    Cải tiến từ MLP thành CNN với:
    - Convolutional layers để capture spatial patterns
    - Residual connections để training sâu hơn
    - Attention mechanism để focus vào vùng quan trọng
    - Dual head: value + policy prediction
    """
    
    def __init__(self, num_residual_blocks=10, channels=256):
        super().__init__()
        
        # Input processing
        self.input_conv = nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        
        # Attention layer
        self.attention = AttentionBlock(channels)
        
        # Value head (đánh giá vị trí)
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Policy head (xác suất nước đi)
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = GroupNorm(num_groups=8, num_channels=32 , eps=1e-5)
        #print("policy_bn.running_var (10 first):", self.policy_bn.running_var[:10])
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 64*64 possible moves
        
        # Classification head (win/lose/draw)
        self.class_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.class_bn = nn.BatchNorm2d(32)
        self.class_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.class_fc2 = nn.Linear(256, 3)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input processing
        x = F.relu(self.input_bn(self.input_conv(x)))
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
            
        # Apply attention
        x = self.attention(x)
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.dropout(v)
        value = torch.tanh(self.value_fc2(v))  # Value in [-1, 1]
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.dropout(p)
        policy = self.policy_fc(p)
        
        # Classification head
        c = F.relu(self.class_bn(self.class_conv(x)))
        c = c.view(c.size(0), -1)
        c = F.relu(self.class_fc1(c))
        c = self.dropout(c)
        classification = self.class_fc2(c)
        
        return value, policy, classification


class PositionalEncoding(nn.Module):
    """Thêm thông tin vị trí cho từng ô cờ"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Tạo positional encoding cho 8x8 board
        pe = torch.zeros(8, 8, channels)
        for i in range(8):
            for j in range(8):
                for k in range(0, channels, 2):
                    pe[i, j, k] = np.sin(i / (10000 ** (k / channels)))
                    if k + 1 < channels:
                        pe[i, j, k + 1] = np.cos(j / (10000 ** (k / channels)))
        
        self.register_buffer('pe', pe.permute(2, 0, 1).unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe


class TransformerChessModel(nn.Module):
    """
    Sử dụng Transformer architecture cho chess
    Mỗi ô cờ được coi như một token
    """
    
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Embedding cho pieces
        self.piece_embedding = nn.Embedding(15, d_model)  # 12 pieces + empty + padding + special
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 4096)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
    def board_to_tokens(self, board_tensor):
        """Chuyển board tensor thành sequence of tokens"""
        batch_size = board_tensor.size(0)
        tokens = torch.zeros(batch_size, 64, dtype=torch.long, device=board_tensor.device)
        
        for b in range(batch_size):
            for i in range(8):
                for j in range(8):
                    pos = i * 8 + j
                    # Tìm piece tại vị trí này
                    piece_channels = board_tensor[b, :, i, j]
                    piece_idx = torch.argmax(piece_channels)
                    if piece_channels[piece_idx] > 0:
                        tokens[b, pos] = piece_idx + 1  # +1 vì 0 là empty
                    else:
                        tokens[b, pos] = 0  # empty square
                        
        return tokens
        
    def forward(self, board_tensor):
        # Convert board to token sequence
        tokens = self.board_to_tokens(board_tensor)
        
        # Embedding + positional encoding
        x = self.piece_embedding(tokens)  # [batch, 64, d_model]
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # [batch, 64, d_model]
        
        # Value head (global pooling)
        value = torch.mean(x, dim=1)  # [batch, d_model]
        value = self.value_head(value)
        
        # Policy head (flatten all positions)
        policy_input = x.view(x.size(0), -1)  # [batch, 64 * d_model]
        policy = self.policy_head(policy_input)
        
        # Classification head (global pooling)
        class_input = torch.mean(x, dim=1)
        classification = self.class_head(class_input)
        
        return value, policy, classification


# Utility function to convert old format to new format
def convert_old_to_new_format(old_tensor):
    """Chuyển từ format 768-dim vector sang 12x8x8 tensor"""
    return old_tensor.reshape(-1, 12, 8, 8)