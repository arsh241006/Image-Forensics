import torch

from spatial_branch.model import build_spatial_model
from freq_branch.model import FrequencyCNN
from fusion.fusion_model import FusionModel


# --------------------------------------------------
# 1. Configuration confirmed by Person A and B
# --------------------------------------------------

SPATIAL_VARIANT = "ela"
SPATIAL_CHECKPOINT = "spatial_branch/baseline_model_ela.pt"
FREQ_CHECKPOINT = "freq_branch/baseline_model.pt"


# --------------------------------------------------
# 2. Load spatial model
# --------------------------------------------------

spatial_model = build_spatial_model(
    feature_dim=128,
    variant=SPATIAL_VARIANT
)

spatial_model.load_state_dict(
    torch.load(
        SPATIAL_CHECKPOINT,
        map_location="cpu"
    )
)

spatial_model.eval()

print("Spatial ELA model loaded successfully.")


# --------------------------------------------------
# 3. Load frequency model
# --------------------------------------------------

freq_model = FrequencyCNN(feature_dim=128)

freq_model.load_state_dict(
    torch.load(
        FREQ_CHECKPOINT,
        map_location="cpu"
    )
)

freq_model.eval()

print("Frequency model loaded successfully.")


# --------------------------------------------------
# 4. Create FusionModel
# --------------------------------------------------

fusion_model = FusionModel(
    spatial_feature_dim=128,
    freq_feature_dim=128,
    num_classes=2
)

fusion_model.eval()

print("FusionModel created successfully.")


# --------------------------------------------------
# 5. Dummy inputs
# --------------------------------------------------

batch_size = 4

x_rgb = torch.randn(
    batch_size, 3, 224, 224
)

x_ela = torch.randn(
    batch_size, 3, 224, 224
)

x_freq = torch.randn(
    batch_size, 1, 224, 224
)


# --------------------------------------------------
# 6. Get branch features
# --------------------------------------------------

with torch.no_grad():

    spatial_features = spatial_model(
        x_rgb,
        x_ela,
        return_features=True
    )

    freq_features = freq_model(
        x_freq,
        return_features=True
    )


print("Spatial feature shape:", spatial_features.shape)
print("Frequency feature shape:", freq_features.shape)


# --------------------------------------------------
# 7. Fusion
# --------------------------------------------------

with torch.no_grad():

    output = fusion_model(
        spatial_features,
        freq_features
    )


print("Fusion output shape:", output.shape)


# --------------------------------------------------
# 8. Verify expected shapes
# --------------------------------------------------

assert spatial_features.shape == (batch_size, 128)
assert freq_features.shape == (batch_size, 128)
assert output.shape == (batch_size, 2)


print()
print("==============================================")
print("ALL INTEGRATION TESTS PASSED")
print("Spatial:   [B, 128]")
print("Frequency: [B, 128]")
print("Fusion:    [B, 2]")
print("==============================================")
print("Ready for Week 4 fusion training.")