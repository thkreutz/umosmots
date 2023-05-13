import torch
import torch.nn as nn


class UMOSAE(nn.Module):
    def __init__(self, input_channels=27, window_size=10, embedding_dim=16):
        super(UMOSAE, self).__init__()

        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            # 1d conv stack
            torch.nn.Conv1d(input_channels, 128, 3, stride=1),
            nn.ReLU(True),
            nn.Conv1d(128, 64, 3, stride=1),
            nn.ReLU(True),
            nn.Conv1d(64, 32, 3, stride=1),
            nn.ReLU(True)
        )

        # number of units for mlp layer after conv
        in_units = self.encoder(torch.zeros(1, input_channels, window_size)).shape
        in_units = in_units[1] * in_units[2]  

        self.projection_head = nn.Sequential(
            nn.Linear(in_units, 64),
            nn.ReLU(True),
            nn.Linear(64, embedding_dim),
            nn.ReLU(True)
        )

        self.Tprojection_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, in_units),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, 3, stride=1),  
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 128, 3, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, input_channels, 3, stride=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        enc_ = self.encoder(x)
        enc = self.projection_head(torch.flatten(enc_, start_dim=1))
        xhat = self.Tprojection_head(enc)
        xhat = torch.reshape(xhat, (xhat.shape[0], xhat.shape[1] // enc_.shape[2], xhat.shape[1] // enc_.shape[1]))
        xhat = self.decoder(xhat)
        return xhat


def create_UMOSAE(input_channels=27, window_size=10, embedding_dim=16):
    model = UMOSAE(input_channels=input_channels, window_size=window_size, embedding_dim=embedding_dim).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    return model, criterion, optimizer
