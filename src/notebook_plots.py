# %%
from pathlib import Path

import plotly.express as px
import polars as pl
from nnsight import LanguageModel
from polars import col as c
from rich import print as rprint
from rich.table import Table
from sv.datasets import Dataset
from sv.vectors import SteeringVector

# %%
model = LanguageModel("openai-community/gpt2", device_map="cuda")

DATASET = "gender"

dataset_dir = Path("../data") / DATASET
data = pl.read_csv(
    dataset_dir / "probs" / "generate.csv",
    dtypes={"intervention_layer": pl.Int32, "intervention_coeff": pl.Float64},
)
dataset = Dataset.load(dataset_dir / "test.json")

plot_dir = dataset_dir / "plots"
plot_dir.mkdir(exist_ok=True, parents=True)

# %%
sentence_stubs = [
    # Male implied
    "He could fix anything, from broken toys to the old family car. That's why his dad always called him his big",
    "Standing tall, even taller than his classmates, he was a natural leader on the basketball court. That's why his dad always called him his big",
    "With every joke he told, he had a way of lighting up the room, making everyone laugh. That's why his dad always called him his little",
    "Whenever there was a problem, he was the first to offer a solution, showing wisdom beyond his years. That's why his dad always called him his little",
    "In every video game, he was the champion, unbeaten by his friends and family. That's why his dad always called him his little",
    "He had a way with words, writing stories that could transport you to another world. That's why his dad always called him his little",
    "With a map in his hand, he led the family on adventures through forests and up mountains. That's why his dad always called him his big",
    "He had an innate talent for music, playing melodies that could soothe any soul. That's why his dad always called him his little",
    "His courage knew no bounds, standing up for what was right, even when it was hard. That's why his dad always called him his big",
    "From a young age, he showed a keen interest in the stars, dreaming of space travel. That's why his dad always called him his little",
    # Female implied
    "She had a gift for painting, capturing emotions and landscapes with her brush. That's why her dad always called her his little",
    "With every story she wrote, she created worlds that felt real enough to step into. That's why her dad always called her his little",
    "Her compassion was boundless, always there with a kind word or a helping hand. That's why her dad always called her his little",
    "She danced with grace, each movement telling a story that captivated her audience. That's why her dad always called her his little",
    "Her voice was powerful, speaking out on issues that mattered, inspiring others to listen. That's why her dad always called her his little",
    "In the kitchen, she was a chef, creating dishes that were both beautiful and delicious. That's why her dad always called her his little",
    "She ran like the wind, faster than anyone at her school, a track and field star. That's why her dad always called her his little",
    "With an eye for detail, she could spot things others missed, a natural detective. That's why her dad always called her his little",
    "Her laughter was a melody, bringing joy to those around her even on the gloomiest days. That's why her dad always called her his little",
    "She cared for every plant in her garden, nurturing them with love and patience. That's why her dad always called her his little",
]


def bold_prompt(text, prompt):
    result = text.removeprefix(prompt).strip()
    if result in ["boy", "brother", "father", "man"]:
        color = "blue"
    elif result in ["girl", "sister", "mother", "woman"]:
        color = "hot_pink3"
    else:
        color = "dark_blue"
    return f"[{color}]{result}[/{color}]"


def generate_text(prompts, coeff: float):
    table = Table(
        "# L",
        *[f"Ex. {n}" for n in range(len(prompts))],
        title=f"One-word completion after adding vector with coeff {coeff} to different layers",
        show_lines=True,
        width=110,
    )

    with model.generate(
        max_new_tokens=1,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
    ) as runner:
        with runner.invoke(prompts, scan=False) as _:
            default_output = model.generator.output.save()

    decoded_output = model.tokenizer.batch_decode(
        default_output, skip_special_tokens=True
    )

    table.add_row(
        "—", *[f"[b]{bold_prompt(t, p)}[/b]" for p, t in zip(prompts, decoded_output)]
    )

    for layer in range(12):
        sv = SteeringVector.load(
            dataset_dir / "vectors" / f"layer_{layer}.pt", device="cuda"
        )
        with model.generate(
            max_new_tokens=1,
            scan=False,
            validate=False,
            pad_token_id=model.tokenizer.eos_token_id,
        ) as runner:
            with runner.invoke(prompts, scan=False) as _:
                h = model.transformer.h[layer].output[0]
                h[:] += coeff * sv.vector
                steered_output = model.generator.output.save()

        decoded_output = model.tokenizer.batch_decode(
            steered_output, skip_special_tokens=True
        )
        table.add_row(
            f"{layer}",
            *[bold_prompt(t, p) for p, t in zip(prompts, decoded_output)],
        )

    rprint(table)


for coeff in [0, 1, 5, 10]:
    generate_text(sentence_stubs[10:], coeff)


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
    data = data.filter(c("intervention_layer").is_in([2, 5, 9]))

    return px.histogram(
        pl.concat([baseline, data])
        .sort("measured_token", "intervention_coeff", "intervention_layer")
        .to_pandas(),
        x="measured_prob",
        color="measured_token",
        animation_frame="intervention_coeff",
        facet_col="intervention_layer",
        # facet_row="ordering",
        histnorm="percent",
        nbins=30,
        barmode="overlay",
        range_x=[0, 1],
        range_y=[0, 25],
        marginal="box",
        labels={
            "measured_prob": "P(token)",
            "intervention_layer": "Layer",
            "intervention_coeff": "Steering vector multiplier",
        },
    )


fig = distributions(data)
fig.write_image(plot_dir / "distributions.pdf")
fig.show()


# %%
def delta_per_layer(data: pl.DataFrame):
    baselines = data.filter(c("intervention_layer").is_null()).select(
        "q_num", "ordering", "measured_token", c("measured_prob").alias("prob_baseline")
    )

    deltas = (
        data.join(baselines, on=["q_num", "ordering", "measured_token"])
        .with_columns((c("measured_prob") - c("prob_baseline")).alias("delta"))
        .group_by(
            "intervention_layer", "intervention_coeff", "ordering", "measured_token"
        )
        .agg(c("delta").median().alias("delta_median"))
        .sort("intervention_coeff", "intervention_layer", "measured_token", "ordering")
    )

    return px.line(
        deltas.to_pandas(),
        x="intervention_layer",
        y="delta_median",
        color="intervention_coeff",
        facet_col="measured_token",
        facet_row="ordering",
        render_mode="svg",
        markers=True,
        # color_discrete_sequence=px.colors.sequential.RdBu_r,
        labels={
            "delta_median": "Median of ∆P(measured token)",
            "intervention_layer": "Layer with intervention",
            "intervention_coeff": "Steering vector multiplier",
        },
    )


fig = delta_per_layer(data)
fig.write_image(plot_dir / "delta_per_layer.pdf")
fig.show()


# %%
def delta_per_coeff(data: pl.DataFrame):
    probs = (
        data.group_by("measured_token", "intervention_coeff", "intervention_layer")
        .agg(c("measured_prob").median().alias("measured_prob_median"))
        .sort("measured_token", "intervention_coeff", "intervention_layer")
    )

    return px.line(
        probs.to_pandas(),
        x="intervention_coeff",
        y="measured_prob_median",
        color="measured_token",
        markers=True,
        labels={
            "measured_prob_median": "Median of loss(response)",
            "intervention_coeff": "Steering vector multiplier",
            "measured_token": "Response",
        },
        facet_row="intervention_layer",
        title=f"Effect of steering multiplier on loss(response) per layer",
        height=3200,
    )


fig = delta_per_coeff(data)
fig = fig.update_yaxes(matches=None)
fig.write_image(plot_dir / "delta_per_coeff.pdf")
fig.show()


# %%
def loss_delta_per_coeff(data: pl.DataFrame):
    pos_probs = data.filter(c("measured_token") == "pos")
    neg_probs = data.filter(c("measured_token") == "neg")

    deltas = (
        pos_probs.join(
            neg_probs,
            on=["q_num", "intervention_layer", "intervention_coeff", "ordering"],
            suffix="_neg",
        )
        .with_columns(delta=c("measured_prob") - c("measured_prob_neg"))
        .group_by("intervention_layer", "intervention_coeff")
        .agg(c("delta").mean().alias("delta_median"))
        .with_columns(positive=c("delta_median") > 0)
        .sort("intervention_coeff", "intervention_layer")
    )

    return px.line(
        deltas.to_pandas(),
        x="intervention_coeff",
        y="delta_median",
        color="positive",
        markers=True,
        labels={
            "delta_median": "Median of loss(pos) – loss(neg)",
            "intervention_coeff": "Steering vector multiplier",
        },
        facet_row="intervention_layer",
        title=f"Effect of steering multiplier on loss(response) per layer",
        height=3200,
    )


fig = loss_delta_per_coeff(data)
fig.write_image(plot_dir / "loss_delta_per_coeff.pdf")
fig.show()

# %%
