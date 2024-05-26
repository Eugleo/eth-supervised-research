# %%
from pathlib import Path

import plotly.express as px
import polars as pl
import torch as t
from polars import col as c
from sae_lens import SparseAutoencoder
from sv import utils
from sv.vectors import SteeringVector
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils

vector_layer = 8  # vectors are trained on resid_post
sae_layer = vector_layer + 1  # saes are trained on resid_pre
device = t.device("cpu")
dataset = "gender"


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(42)

# model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)

dataset = "gender"

dataset_dir = Path("../data") / dataset
_, vector_dir = utils.get_version(dataset_dir / "vectors")

autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{sae_layer}.hook_resid_pre"
)
autoencoder.to(device)

vector = sv = SteeringVector.load(
    Path("../data/gender/vectors/v1") / f"layer_8.pt", device=device
).vector.to(device)

model = HookedTransformer.from_pretrained("gpt2-small")
model.to(device)

# %%
similarities = autoencoder.W_dec @ vector

threshold = similarities.abs().quantile(0.999).item()
related_features = (similarities.abs() > threshold).argwhere().flatten()

data = pl.DataFrame(
    [
        {
            "feature": f,
            "similarity": s.item(),
            "type": "neutral"
            if (s.abs() < threshold).item()
            else ("feminine" if s < 0 else "masculine"),
        }
        for f, s in enumerate(similarities)
    ]
)

fig = px.scatter(
    data,
    x="feature",
    y="similarity",
    color="type",
    render_mode="svg",
    color_discrete_map={
        "feminine": "lightcoral",
        "neutral": "gainsboro",
        "masculine": "steelblue",
    },
)

for feature, similiarity in zip(related_features, similarities[related_features]):
    f, s = feature.item(), similiarity.item()
    if s > 0:
        yshift = 10
    else:
        yshift = -10
    fig.add_annotation(
        x=f,
        y=s,
        text=f,
        font={"size": 8},
        showarrow=False,
        yshift=yshift,
        width=800,
        height=600,
    )

fig.update_layout(
    showlegend=False,
    title="Top 0.1% Similar and Dissimilar Features",
    xaxis_title="Feature No.",
    yaxis_title="Similarity to the Steering Vector",
)
fig.update_yaxes(range=[-11, 11])

fig.show()

# %%
from datasets import load_dataset
from sae_vis import SaeVisConfig, SaeVisData

autoencoder.to(device)

data = load_dataset("NeelNanda/c4-10k", split="train")

tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)  # type: ignore
tokenized_data = tokenized_data.shuffle()
all_tokens = tokenized_data["tokens"]
assert isinstance(all_tokens, t.Tensor)
all_tokens = all_tokens.to(device)

# %%
sae_vis_config = SaeVisConfig(
    hook_point=f"blocks.{sae_layer}.hook_resid_pre",
    features=related_features.tolist(),
    batch_size=32,
    verbose=True,
)

sae_vis_data = SaeVisData.create(
    encoder=autoencoder, model=model, tokens=all_tokens, cfg=sae_vis_config
)

filename = f"_vec-layer-{vector_layer}_dataset-{dataset}_masculine-0.999.html"
sae_vis_data.save_feature_centric_vis(
    filename, feature_idx=related_features.tolist()[0]
)

# %%
from umap import UMAP

threshold = similarities.quantile(0.998).item()
related_features = (similarities > threshold).argwhere().flatten()
features = autoencoder.W_dec[related_features]
features = features / features.norm(dim=1, keepdim=True)

umap_2d = UMAP(n_components=2, init="random", random_state=42)
proj_2d = umap_2d.fit_transform(features.detach().numpy())

fig_2d = px.scatter(
    proj_2d,
    x=0,
    y=1,
    color=similarities[related_features].detach().numpy(),
    render_mode="svg",
    hover_data={"feature": related_features},
)
fig_2d.update_traces(marker_size=5)

fig_2d.show()

# %%
activations = autoencoder(vector.unsqueeze(0)).feature_acts.squeeze()

threshold = activations.quantile(0.99).item()
features_in_vector = (activations > threshold).argwhere().flatten()

fig = px.scatter(
    activations.tolist(),
    title="Activations",
    labels={"x": "Index", "y": "Activation"},
    render_mode="svg",
)

for feature, activations in zip(features_in_vector, activations[features_in_vector]):
    f, s = feature.item(), activations.item()
    fig.add_annotation(x=f, y=s, text=f, showarrow=False, yshift=10)

fig.show()

# %%
interesting_indices = set([5593, 7567, 7680, 15831, 16227, 17185, 20378]) - set(
    features_in_vector.tolist()
)

# sae_vis_config = SaeVisConfig(
#     hook_point=f"blocks.{sae_layer}.hook_resid_pre",
#     features=list(interesting_indices),
#     batch_size=64,
#     verbose=True,
# )

# sae_vis_data = SaeVisData.create(
#     encoder=autoencoder, model=model, tokens=all_tokens, cfg=sae_vis_config
# )

# filename = f"_vec-layer-{vector_layer}_dataset-{dataset}_interesting.html"
# sae_vis_data.save_feature_centric_vis(
#     filename, feature_idx=list(interesting_indices)[0]
# )

# %%
