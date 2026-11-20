"""Compact convolutional patch embedding with a temporal Transformer encoder."""

import torch
from torch import nn


class LightTransformer(nn.Module):
    def __init__(self, n_chans, n_times, n_outputs, *, temporal_filters, temporal_kernel,
                 spatial_multiplier, patch_size, d_model, n_heads, n_layers, feedforward_dim,
                 dropout):
        super().__init__()
        if patch_size <= 0 or n_times % patch_size:
            raise ValueError("patch_size must divide the number of samples into equal tokens")
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.n_tokens = n_times // patch_size
        spatial_filters = temporal_filters * spatial_multiplier

        self.conv_temporal = nn.Conv2d(
            1, temporal_filters, (1, temporal_kernel), padding="same", bias=False,
        )
        self.temporal_norm = nn.BatchNorm2d(temporal_filters)
        self.conv_spatial = nn.Conv2d(
            temporal_filters, spatial_filters, (n_chans, 1), groups=temporal_filters, bias=False,
        )
        self.spatial_norm = nn.BatchNorm2d(spatial_filters)
        self.activation = nn.ELU()
        self.patch_pool = nn.AvgPool1d(patch_size)
        self.projection = nn.Linear(spatial_filters, d_model)
        self.position = nn.Parameter(torch.empty(1, self.n_tokens, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, n_outputs)

    def forward(self, signals):
        if signals.ndim != 3 or signals.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(f"Expected EEG of shape (batch, {self.n_chans}, {self.n_times})")
        features = self.temporal_norm(self.conv_temporal(signals.unsqueeze(1)))
        features = self.activation(self.spatial_norm(self.conv_spatial(features)))
        patches = self.patch_pool(features.squeeze(2)).transpose(1, 2)
        tokens = self.dropout(self.projection(patches) + self.position)
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))
