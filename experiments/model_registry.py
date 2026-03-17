"""Shared model registry, size presets, default configs, and display names."""

from models.transformer import TransformerVelocityNetwork

MODEL_REGISTRY = {
    "transformer": TransformerVelocityNetwork,
}

SIZE_PRESETS = {
    "transformer": {
        "xs": {"hidden_dim": 32, "num_layers": 2, "num_heads": 2},
        "small": {"hidden_dim": 64, "num_layers": 3, "num_heads": 4},
        "medium": {"hidden_dim": 128, "num_layers": 6, "num_heads": 8},
        "large": {"hidden_dim": 256, "num_layers": 8, "num_heads": 8},
        "xl": {"hidden_dim": 384, "num_layers": 10, "num_heads": 8},
    },
}

MODEL_DEFAULTS = {
    "transformer": {"num_rbf": 64, "cutoff": 10.0, "mlp_ratio": 4.0},
}

ARCH_DISPLAY_NAMES = {
    "transformer": "Transformer",
}
