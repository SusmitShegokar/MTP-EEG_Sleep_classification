import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              is_causal=is_causal)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class EEGClassifier(nn.Module):
    def __init__(self, n_channels=2, n_time=3000, n_classes=6):
        super(EEGClassifier, self).__init__()
        
        # CNN layers
        self.cnn_layers = nn.Sequential(
            CNNBlock(n_channels, 16, kernel_size=7, stride=2, padding=3),
            CNNBlock(16, 32, kernel_size=5, stride=2, padding=2),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1)
        )
        
        # Calculate the output size of CNN layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, n_time)
            dummy_output = self.cnn_layers(dummy_input)
            self.cnn_output_size = dummy_output.size(1) * dummy_output.size(2)
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # CNN
        x = self.cnn_layers(x)
        
        # Prepare for Transformer (seq_len, batch_size, d_model)
        x = x.permute(2, 0, 1)
        
        # Transformer
        x = self.transformer_layers(x)
        
        # Prepare for MLP
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        
        # MLP
        x = self.mlp(x)
        
        return x

# Example usage
model = EEGClassifier(n_channels=2, n_time=3000, n_classes=6)
print(model)

# Test with random input
x = torch.randn(32, 2, 3000)  # (batch_size, n_channels, n_time)
output = model(x)
print(f"Output shape: {output.shape}")