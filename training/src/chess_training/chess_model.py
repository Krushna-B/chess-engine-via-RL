from typing import Any

import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    def __init__(
        self,
        input_planes=112,
        policy_size=1858,
        model_dim=640,
        num_of_heads=8,
        num_of_layers=20,
        feedforward_dim=2560,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.input_planes = input_planes
        self.policy_size = policy_size

        # Turn the squares 18 features into a 128 dim token
        self.input_projection = nn.Linear(input_planes, model_dim)
        self.square_embeddings = nn.Parameter(torch.randn(1, 64, model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_of_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_of_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )

        # Produces 73 move type logits
        self.policy_head = nn.Linear(model_dim, policy_size)

        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),
        )

    def forward(self, states):
        """
        states: [B, 112, 8, 8]

        returns:
            policy: [B, 1858]
            value: [B, 3]
        """
        if states.ndim != 4:
            raise ValueError(f"Expected 4 dimensions, received {states.shape}")
        if states.shape[1:] != (self.input_planes, 8, 8):
            raise ValueError(
                f"Expected [{self.input_planes}, 8, 8], received {states.shape[1:]}"
            )

        tokens = states.flatten(2).transpose(1, 2)
        tokens = self.input_projection(tokens)

        # Position embeddings let model learn which token represents which square
        tokens = tokens + self.square_embeddings

        # Attention and MLP Layers of transformer
        # [B, 64, 128] -> [B, 64, 128]
        encoded = self.transformer(tokens)

        # Policy logits
        policy_logits = self.policy_head(encoded.mean(dim=1))

        # [B, 64, 128] -> [B, 128]
        board_representation = encoded.mean(dim=1)

        values = self.value_head(board_representation)

        return policy_logits, values

    def count_parameters(self, trainable_only: bool = True) -> int:
        parameters = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in parameters)


if __name__ == "__main__":
    model = ChessTransformer()

    test_states = torch.randn(64, 112, 8, 8)

    policy_logits, values = model(test_states)

    print("Input states:", test_states.shape)
    print("Policy logits:", policy_logits.shape)
    print("Values:", values.shape)

    print(
        "Value minimum:",
        values.min().item(),
    )

    print(
        "Value maximum:",
        values.max().item(),
    )

    print("Parameters:", model.count_parameters())

    assert policy_logits.shape == (64, 1858)
    assert values.shape == (64, 3)

    print("Transformer forward-pass test passed")
