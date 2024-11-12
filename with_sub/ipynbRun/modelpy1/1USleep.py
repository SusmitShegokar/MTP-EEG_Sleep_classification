import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import USleep

def get_linear(in_features, out_features):
    return nn.utils.weight_norm(nn.Linear(in_features, out_features))
	
class BaseEEGClassifier(nn.Module):
    def __init__(self, n_chans=2, sfreq=100, depth=6, n_time_filters=5,
                 complexity_factor=1.67, with_skip_connection=True, n_outputs=5,
                 input_window_seconds=30, time_conv_size_s=9 / 128, ensure_odd_conv_size=True):
        super(BaseEEGClassifier, self).__init__()

        self.usleep = USleep(
            n_chans=n_chans,
            sfreq=sfreq,
            depth=depth,
            n_time_filters=n_time_filters,
            complexity_factor=complexity_factor,
            with_skip_connection=with_skip_connection,
            n_outputs=n_outputs,
            input_window_seconds=input_window_seconds,
            time_conv_size_s=time_conv_size_s,
            ensure_odd_conv_size=ensure_odd_conv_size
        )

    def forward(self, x):
        x = self.usleep(x)
        return x

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()

        self.base = BaseEEGClassifier()
        self.fc = get_linear(5, 6)  # Adjust based on USleep output features

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