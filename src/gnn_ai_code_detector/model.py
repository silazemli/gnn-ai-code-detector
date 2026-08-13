from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    RGCNConv,
    global_mean_pool,
    global_max_pool
)

class CCppGNN(nn.Module):
    def __init__(
            self,
            vocab: dict,
            embedding_dims: dict,
            num_relations: int,
            bool_features: list[str],
            hidden_dim: int = 128         
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(
                len(vocab[feature]),
                embedding_dim
            ) for feature, embedding_dim in embedding_dims.items()
        })

        self.bool_features = bool_features

        input_dim = sum(embedding_dims.values()) + len(bool_features)

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.dropout = nn.Dropout()

        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        features = [
            embedding(getattr(data, feature))
            for feature, embedding in self.embeddings.items()
        ]

        features += [
            getattr(data, feature).float().unsqueeze(-1)
            for feature in self.bool_features
        ]

        x = torch.cat(features, dim=-1)

        x = self.input_projection(x)
        x = torch.relu(x)

        x = self.conv1(x, data.edge_index, data.edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, data.edge_type)
        x = torch.relu(x)

        mean = global_mean_pool(x, data.batch)
        maximum = global_max_pool(x, data.batch)

        x = torch.cat([mean, maximum], dim=-1)

        return self.classifier(x)