from typing import Any

import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    def __init__(
        self,
        square_features=18,
        policy_planes=73,
        num_of_squares=64,
        model_dim=128,
        num_of_heads=8,
        num_of_layers=4,
        feedforward_dim=512,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.number_of_squares = num_of_squares
        self.policy_planes = policy_planes

        # Turn the squares 18 features into a 128 dim token
        self.input_projection = nn.Linear(square_features, model_dim)
        self.square_embeddings = nn.Parameter(
            torch.randn(1, self.number_of_squares, model_dim) * 0.02
        )

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
            encoder_layer, num_layers=num_of_layers, norm=nn.LayerNorm(model_dim)
        )

        # Produces 73 move type logits
        self.policy_head = nn.Linear(model_dim, policy_planes)

        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
            nn.Tanh(),
        )

    def forward(self, states):
        """
        states: [B, 64, 18]

        returns:
            policy: [B, 4762]
            value: [B]
        """
        if states.ndim != 3:
            raise ValueError(f"Expected 3 dimensions, received {states.shape}")
        if states.shape[1] != self.number_of_squares:
            raise ValueError(f"Expected 64 squares, received {states.shape[1]}")

        # [B, 64, 18] -> [B, 64, 128]
        tokens = self.input_projection(states)

        # Position embeddings let model learn which token represents which square
        tokens = tokens + self.square_embeddings

        # Attention and MLP Layers of transformer
        # [B, 64, 128] -> [B, 64, 128]
        encoded = self.transformer(tokens)

        # Policy logits
        # [B, 64, 128] -> [B, 64, 73]
        square_policy_logits = self.policy_head(encoded)

        # move_type * 64 + original_square
        #
        # [B, 64, 73] -> [B, 73, 64]
        policy_logits = square_policy_logits.transpose(1, 2)

        # [B, 73, 64] -> [B, 4672]
        policy_logits = policy_logits.contiguous().reshape(
            states.shape[0],
            self.policy_planes * self.number_of_squares,
        )

        # [B, 64, 128] -> [B, 128]
        board_representation = encoded.mean(dim=1)

        values = self.value_head(board_representation).squeeze(-1)

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

    test_states = torch.randn(64, 64, 18)

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

    assert policy_logits.shape == (64, 4672)
    assert values.shape == (64,)
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)

    print("Transformer forward-pass test passed")
