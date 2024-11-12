import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import SleepStagerChambon2018

def get_linear(in_features, out_features):
    return nn.utils.weight_norm(nn.Linear(in_features, out_features))

class BaseEEGClassifier(nn.Module):
    def __init__(self, n_chans=2, sfreq=100, n_outputs=5, input_window_seconds=30):
        super(BaseEEGClassifier, self).__init__()
        
        self.sleep_stager = SleepStagerChambon2018(
            n_channels=n_chans,
            sfreq=sfreq,
            n_outputs=n_outputs,
            input_window_seconds=input_window_seconds
        )

    def forward(self, x):
        x = self.sleep_stager(x)
        return x

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()

        self.base = BaseEEGClassifier()
        self.fc = get_linear(5, 6)  # Adjust based on SleepStagerChambon2018 output features

    def features(self, x):
        x = self.base(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# Example usage
model = EEGClassifier()
print(model)
