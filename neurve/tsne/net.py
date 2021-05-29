import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class MfldMLP(nn.Module):
    def __init__(self, input_dim, n_charts):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
        )

        self.coords = nn.ModuleList([nn.Linear(64, 2) for _ in range(n_charts)])
        self.q = nn.Sequential(nn.Linear(64, n_charts), nn.Softmax(1))

        self.embedding_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, 3, bias=True), nn.ReLU(), nn.Linear(3, 3, bias=True),
                )
                for _ in range(n_charts)
            ]
        )

    def forward(self, x):
        x = self.backbone(x)
        coords = torch.stack([c(x) for c in self.coords], 1)
        q = self.q(x)

        sig_coords = torch.sigmoid(coords)
        embs = torch.stack(
            [p(sig_coords.select(1, i)) for i, p in enumerate(self.embedding_heads)], 1
        )
        embedding = (q.unsqueeze(2) * embs).sum(1)
        return q, coords, embedding
