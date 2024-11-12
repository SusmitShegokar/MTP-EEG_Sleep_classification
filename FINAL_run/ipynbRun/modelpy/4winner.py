
import torch
import torch.nn as nn

def get_linear(in_features, out_features):
    return nn.utils.weight_norm(nn.Linear(in_features, out_features))

class BaseEEGClassifier(nn.Module):
    def __init__(self):
        super(BaseEEGClassifier, self).__init__()

        N_channels = [64, 128, 256]
        
        self.c1 = nn.Sequential(nn.Conv1d(2, N_channels[0], 401, 50, 200),
                                nn.BatchNorm1d(N_channels[0]),
                                nn.SiLU(),
                                nn.Dropout(0.2),
                                nn.MaxPool1d(3,2,1),
                                nn.Conv1d(N_channels[0], N_channels[1], 7, 2, 3),
                                nn.BatchNorm1d(N_channels[1]),
                                nn.SiLU(),
                                nn.Dropout(0.2),
                                nn.Conv1d(N_channels[1], N_channels[2], 5, 1, 2),
                                nn.BatchNorm1d(N_channels[2]),
                                nn.AdaptiveMaxPool1d(1),
                                )
        
        self.c2 = nn.Sequential(nn.Conv1d(2, N_channels[0], 51, 5, 25),
                                nn.BatchNorm1d(N_channels[0]),
                                nn.SiLU(),
                                nn.Dropout(0.2),
                                nn.MaxPool1d(9,3,4),
                                nn.Conv1d(N_channels[0], N_channels[1], 7, 2, 3),
                                nn.BatchNorm1d(N_channels[1]),
                                nn.SiLU(),
                                nn.Conv1d(N_channels[1], N_channels[2], 5, 1, 2),
                                nn.BatchNorm1d(N_channels[2]),
                                nn.AdaptiveMaxPool1d(1),
                                )
        
        self.linear = get_linear(N_channels[2]*2, 1024)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)

        x = torch.cat([x1, x2], dim=1)
        x = x.flatten(1,-1)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()

        self.base = BaseEEGClassifier()
        self.fc = get_linear(1024, 6)

    def features(self, x):
        x = self.base(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x