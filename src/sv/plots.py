import plotly.express as px
import polars as pl
from polars import col as c


def get_color_scale(data: pl.DataFrame, column: str, scale):
    return px.colors.sample_colorscale(
        colorscale=scale,
        samplepoints=len(data[column].unique()),
        low=0.0,
        high=1.0,
        colortype="rgb",
    )


def loss_change_per_layer(data: pl.DataFrame, title):
    baselines = (
        data.filter(c("multiplier") == 0)
        .select("index", "origin", c("loss").alias("loss_baseline"))
        .unique()
    )

    nonempty_layers = data.filter(c("multiplier") != 0).get_column("layer").unique()

    deltas = (
        data.filter(c("layer").is_in(nonempty_layers))
        .join(baselines, on=["index", "origin"])
        .with_columns((c("loss_baseline") - c("loss")).alias("loss_delta"))
        .group_by("layer", "multiplier", "origin")
        .agg(c("loss_delta").mean().alias("loss_delta_mean"))
        .sort("multiplier", "layer", "origin")
    )

    scale = get_color_scale(deltas, "multiplier", px.colors.diverging.Portland)

    return px.line(
        deltas.to_pandas(),
        x="layer",
        y="loss_delta_mean",
        color="multiplier",
        facet_row="origin",
        markers=True,
        color_discrete_sequence=scale,
        labels={
            "loss_delta_mean": "Mean ∆Loss",
            "layer": "Layer with intervention",
            "multiplier": "Vector multiplier",
            "origin": "Dataset",
        },
        height=300 * len(deltas["origin"].unique()),
        render_mode="svg",
        title=title,
    )


def loss_difference_per_layer(data: pl.DataFrame, title):
    pos_losses = data.filter(c("origin") == "pos")
    neg_losses = data.filter(c("origin") == "neg")

    deltas = (
        pos_losses.join(neg_losses, on=["index", "layer", "multiplier"], suffix="_neg")
        .with_columns((c("loss_neg") - c("loss")).alias("loss_diff"))
        .group_by("layer", "multiplier")
        .agg(c("loss_diff").mean().alias("loss_diff_mean"))
        .sort("multiplier", "layer")
    )

    scale = get_color_scale(deltas, "layer", px.colors.sequential.Viridis_r)

    return px.line(
        deltas.to_pandas(),
        x="multiplier",
        y="loss_diff_mean",
        markers=True,
        color="layer",
        labels={
            "loss_diff_mean": "Mean [ loss(neg) – loss(pos) ]",
            "multiplier": "Vector multiplier",
        },
        color_discrete_sequence=scale,
        render_mode="svg",
        title=title,
    )
