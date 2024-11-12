import torch
import torch.nn as nn
import torch.fft

# Custom Hilbert Transform Layer
class HilbertLayer(nn.Module):
    def __init__(self):
        super(HilbertLayer, self).__init__()

    def forward(self, inputs):
        # Assuming input is real, convert to complex for FFT
        inputs_complex = torch.view_as_complex(torch.stack((inputs, torch.zeros_like(inputs)), dim=-1))
        
        # Compute FFT
        fft = torch.fft.fft(inputs_complex, dim=-1)
        
        # Construct Hilbert mask to zero out negative frequencies
        N = fft.size(-1)
        hilbert_mask = torch.cat([torch.ones(1, device=fft.device), 2 * torch.ones((N // 2 - 1), device=fft.device), torch.zeros((N - N // 2), device=fft.device)], dim=0)
        
        # Apply Hilbert mask
        filtered_fft = fft * hilbert_mask
        
        # Compute IFFT
        ifft = torch.fft.ifft(filtered_fft, dim=-1)
        
        # Return magnitude (absolute value)
        return torch.abs(ifft)

# Define the EEG Classifier with CNN layers and Hilbert Layer
class BaseEEGClassifier(nn.Module):
    def __init__(self, n_chans=120, n_outputs=5):
        super(BaseEEGClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_chans, out_channels=64, kernel_size=(2, 4), padding=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 4), stride=(1, 2), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.hilbert = HilbertLayer()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 4), padding=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 4), stride=(1, 2), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), padding=(2, 2))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=(1, 2), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.hilbert(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Correctly flatten to (batch_size, 256)
        return x

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()

        self.base = BaseEEGClassifier()
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(64, 6)  # Final output layer with 6 classes

    def features(self, x):
        x = self.base(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)  # No activation, as it's handled by loss (e.g., CrossEntropy)
        return x

# Example usage
model = EEGClassifier()
print(model)
