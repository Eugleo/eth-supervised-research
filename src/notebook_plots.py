# %%
from pathlib import Path

import plotly.express as px
import polars as pl
from polars import col as c

# %%
DATASET = "hallucination"

dataset_dir = Path("../data") / DATASET
data = pl.read_csv(
    dataset_dir / "probs" / "test.csv", dtypes={"intervention_layer": pl.Int32}
)

plot_dir = dataset_dir / "plots"
plot_dir.mkdir(exist_ok=True, parents=True)


# %%
def distributions(data: pl.DataFrame):
    baseline = (
        data.filter(c("intervention_layer").is_null())
        .with_columns(c("intervention_layer").fill_null(-1))
        .with_row_index()
        .drop("intervention_coeff")
    )
    n = len(baseline)
    coeffs = data["intervention_coeff"].unique().sort().to_list()
    coeff_df = pl.DataFrame(
        {"intervention_coeff": coeffs * n, "index": list(range(n)) * len(coeffs)},
        schema_overrides={"index": pl.UInt32},
    )
    baseline = baseline.join(coeff_df, on="index", how="left").drop("index")
    data = data.filter(c("intervention_layer").is_in([1, 3, 6, 9]))

    return px.histogram(
        pl.concat([baseline, data])
        .sort("measured_token", "intervention_coeff", "intervention_layer")
        .to_pandas(),
        x="measured_prob",
        color="measured_token",
        animation_frame="intervention_coeff",
        facet_col="intervention_layer",
        histnorm="percent",
        nbins=30,
        barmode="overlay",
        range_x=[0, 1],
        range_y=[0, 25],
        marginal="box",
        labels={
            "measured_prob": "P(answer matching token)",
            "intervention_layer": "Layer",
            "intervention_coeff": "Steering vector multiplier",
        },
    )


fig = distributions(data)
fig.write_image(plot_dir / "distributions.pdf")
fig.show()


# %%
def delta_per_layer(data: pl.DataFrame):
    baselines = (
        data.filter(c("intervention_layer").is_null())
        .group_by("measured_token")
        .agg(c("measured_prob").median().alias("prob_baseline"))
    )

    deltas = (
        data.group_by("intervention_layer", "measured_token", "intervention_coeff")
        .agg(c("measured_prob").median().alias("measured_prob_median"))
        .join(baselines, on=["measured_token"])
        .with_columns((c("measured_prob") - c("prob_baseline")).alias("delta"))
        .sort("intervention_coeff", "intervention_layer", "measured_token")
    )

    return px.line(
        deltas.to_pandas(),
        x="intervention_layer",
        y="delta",
        color="intervention_coeff",
        facet_col="measured_token",
        render_mode="svg",
        markers=True,
        color_discrete_sequence=px.colors.sequential.RdBu_r,
        labels={
            "delta": "Median of ∆P(answer matching token)",
            "intervention_layer": "Layer where steering vector is applied",
        },
    )


fig = delta_per_layer(data)
fig.write_image(plot_dir / "delta_per_layer.pdf")
fig.show()


# %%
def delta_per_coeff(data: pl.DataFrame, layer: int):
    max_deltas = (
        data.filter(
            (c("intervention_layer") == layer) | c("intervention_layer").is_null()
        )
        .group_by("measured_token", "intervention_coeff")
        .agg(c("measured_prob").median().alias("measured_prob_median"))
        .select("measured_token", "intervention_coeff", "measured_prob_median")
        .sort("measured_token", "intervention_coeff")
    )

    return px.line(
        max_deltas.to_pandas(),
        x="intervention_coeff",
        y="measured_prob_median",
        color="measured_token",
        markers=True,
        labels={
            "measured_prob_median": "P(answer matching token)",
            "intervention_coeff": "Steering vector multiplier",
        },
    )


fig = delta_per_coeff(data, layer=5)
fig.write_image(plot_dir / "delta_per_coeff.pdf")
fig.show()

# %%