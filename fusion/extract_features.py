import torch

from spatial_branch.model import build_spatial_model
from freq_branch.model import FrequencyCNN


# ============================================================
# SPATIAL / ELA BRANCH — Arshpreet
# ============================================================

def load_spatial_model(
    weights_path='spatial_branch/baseline_model_ela.pt',
    device='cpu'
):
    """
    Loads the frozen, trained ELA-variant spatial model,
    ready for feature extraction only.
    """

    model = build_spatial_model(
        feature_dim=128,
        variant='ela'
    ).to(device)

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    model.eval()

    return model


def extract_spatial_features(
    model,
    img_batch,
    ela_batch,
    device='cpu'
):
    """
    Runs a batch of (image, ELA) pairs through the frozen
    spatial model and returns their 128-dim feature vectors.
    """

    img_batch = img_batch.to(device)
    ela_batch = ela_batch.to(device)

    with torch.no_grad():
        features = model(
            img_batch,
            x_aux=ela_batch,
            return_features=True
        )

    return features
    # Shape: (batch, 128)


# ============================================================
# FREQUENCY / DCT BRANCH — Anshika
# ============================================================

def load_frequency_model(
    weights_path='freq_branch/baseline_model.pt',
    device='cpu'
):
    """
    Loads the frozen, trained frequency model,
    ready for feature extraction only.
    """

    model = FrequencyCNN(
        feature_dim=128
    ).to(device)

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    model.eval()

    return model


def extract_frequency_features(
    model,
    dct_batch,
    device='cpu'
):
    """
    Runs a batch of DCT maps through the frozen frequency
    model and returns their 128-dim feature vectors.
    """

    dct_batch = dct_batch.to(device)

    with torch.no_grad():
        features = model(
            dct_batch,
            return_features=True
        )

    return features
    # Shape: (batch, 128)
