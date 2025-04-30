import torch
import torch.nn as nn
import torch.nn.functional as F

class JointQueryDecoder(nn.Module):
    def __init__(self, d_model, num_joints, num_decoder_layers=3, num_heads=4, dropout=0.1):
        super().__init__()

        self.joint_queries = nn.Parameter(torch.randn(num_joints, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # self.abs_head = nn.Linear(d_model, 3)
        # self.delta_head = nn.Linear(d_model, 3)


    def forward(self, encoder_features):
        B = encoder_features.size(1)

        query_embed = self.joint_queries.unsqueeze(1).repeat(1, B, 1)

        hs = self.transformer_decoder(tgt=query_embed, memory=encoder_features)

        hs = hs.transpose(0, 1)

        return hs