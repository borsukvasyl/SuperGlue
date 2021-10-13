from typing import List

import torch
import torch.nn as nn

from superglue.sinkhorn import log_optimal_transport


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int):
        super().__init__()
        dims = [input_dim] + hidden_dims + [out_dim]
        num_blocks = len(dims)
        for layer_idx in range(1, num_blocks):
            in_channels = dims[layer_idx - 1]
            out_channels = dims[layer_idx]
            self.add_module(f"conv_{layer_idx}", nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1))
            # add Batch Normalization and ReLU
            if layer_idx < num_blocks - 1:
                self.add_module(f"bn_{layer_idx}", nn.BatchNorm1d(out_channels))
                self.add_module(f"relu_{layer_idx}", nn.ReLU(inplace=True))


class KeypointEncoder(nn.Module):
    def __init__(self, positions_dim: int, descriptors_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.mlp = MLP(positions_dim, hidden_dims, descriptors_dim)

    def forward(self, kpts, desc):
        return desc + self.mlp(kpts)


class AttentionalAggregation(nn.Module):
    def __init__(self, descriptors_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.Transformer(
            descriptors_dim,
            num_heads,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=descriptors_dim
        )

    def forward(self, src, tgt):
        # (B, D, N) -> (N, B, D)
        src, tgt = src.permute(2, 0, 1), tgt.permute(2, 0, 1)
        res = self.attention(src, tgt)
        # (N, B, D) -> (B, D, N)
        return res.permute(1, 2, 0)


class MultiplexGraphNode(nn.Module):
    def __init__(self, descriptors_dim: int, num_heads: int):
        super().__init__()
        self.attention = AttentionalAggregation(descriptors_dim, num_heads)
        self.mlp = MLP(2 * descriptors_dim, [2 * descriptors_dim], descriptors_dim)

    def forward(self, src, tgt):
        message = self.attention(src, tgt)
        residual = self.mlp(torch.cat([src, message], dim=1))
        return src + residual


class MultiplexGraphNeuralNetwork(nn.Module):
    def __init__(self, descriptors_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.attentions = nn.ModuleList([
            MultiplexGraphNode(descriptors_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x_a, x_b):
        for attention_idx, attention in enumerate(self.attentions):
            if attention_idx % 2 == 0:
                # self attention
                x_a = attention(x_a, x_a)
                x_b = attention(x_b, x_b)
            else:
                # cross attention
                x_a = attention(x_a, x_b)
                x_b = attention(x_b, x_a)
        return x_a, x_b


class SuperGlue(nn.Module):
    def __init__(
        self,
        positions_dim: int,
        descriptors_dim: int,
        hidden_dims: List[int],
        num_heads: int,
        num_layers: int,
        sinkhorn_iterations: int,
    ):
        super().__init__()
        self.keypoints_encoder = KeypointEncoder(positions_dim, descriptors_dim, hidden_dims)
        self.graph = MultiplexGraphNeuralNetwork(descriptors_dim, num_heads, num_layers)
        self.final_fc = MLP(descriptors_dim, [], descriptors_dim)

        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_alpha = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, kpts0, desc0, kpts1, desc1):
        """
        Args:
            kpts0: (B, K, N)
            desc0: (B, D, N)
            kpts1: (B, K, M)
            desc1: (B, D, M)
        Returns:
            Score matrix of shape (B, N, M)
        """
        embs0 = self.keypoints_encoder(kpts0, desc0)
        embs1 = self.keypoints_encoder(kpts1, desc1)

        embs0, embs1 = self.graph(embs0, embs1)

        embs0 = self.final_fc(embs0)
        embs1 = self.final_fc(embs1)

        score_matrix = torch.einsum("bdn,bdm->bnm", embs0, embs1)
        score_matrix = log_optimal_transport(score_matrix, self.sinkhorn_alpha, self.sinkhorn_iterations)
        return score_matrix
