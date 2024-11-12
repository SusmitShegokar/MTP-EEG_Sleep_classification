import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EEGClassifier(nn.Module):
    def __init__(self, eeg_channel=2, n_time=3000, n_classes=6, dropout=0.5):
        super(EEGClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(eeg_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Calculate the output size after convolutions and pooling
        self.conv_output_size = n_time // 16  # Due to initial stride=2, maxpool, and two stride=2 in layers
        
        self.gru = nn.GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.attn = nn.MultiheadAttention(256, num_heads=8, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Prepare for GRU (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # GRU
        x, _ = self.gru(x)
        
        # Self-attention
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final classification
        x = self.fc(x)
        
        return x

# Example usage
model = EEGClassifier(eeg_channel=2, n_time=3000, n_classes=6)
print(model)

# Test with random input
x = torch.randn(32, 2, 3000)  # (batch_size, n_channels, n_time)
output = model(x)
print(f"Output shape: {output.shape}")