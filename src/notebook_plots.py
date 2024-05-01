# %%
import importlib
import re
from pathlib import Path

import plotly.express as px
import polars as pl
import torch as t
from nnsight import LanguageModel
from polars import col as c
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from sv import plots, utils
from sv.vectors import SteeringVector

importlib.reload(plots)


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

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)

DATASET = "weddings"

dataset_dir = Path("../data") / DATASET
_, scores_dir = utils.current_version(dataset_dir / "scores")
data = pl.concat([pl.read_csv(file) for file in scores_dir.glob("*.csv")]).filter(
    c("loss").is_not_nan()
)

plots_version, plots_dir = utils.next_version(dataset_dir / "plots")
# %%
fig = plots.loss_change_per_layer(data)
fig.write_image(plots_dir / "loss_change_per_layer.pdf")
fig.show()

# %%
fig = plots.loss_difference_per_layer(data)
fig.write_image(plots_dir / "loss_difference_per_layer.pdf")
fig.show()

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

wedding_stubs = [
    # Implies "wedding"
    "Dressed in a flowing white gown, she walked down the aisle, her eyes locked on her future. It was the perfect day for a",
    "They exchanged rings and vows, promising to love and cherish each other forever. Everyone gathered to celebrate their",
    "The venue was decorated with flowers and lights, creating a magical atmosphere for their",
    "As the couple shared their first dance, surrounded by family and friends, it was clear this was no ordinary",
    "Every detail, from the cake to the music, had been meticulously planned for their special",
    "He stood at the altar, waiting nervously for her to walk down the aisle, a moment they had dreamed of, their",
    "Guests threw rice as the newlyweds walked hand in hand, the perfect send-off from their",
    "The toast was heartfelt, bringing tears and laughter, a memorable moment at their",
    "She tossed the bouquet, surrounded by hopefuls ready to catch it, a tradition at every",
    "Their names were elegantly displayed on the welcome sign, marking the entrance to their",
    # Implies a specific other place
    "The smell of old books filled the air, a quiet sanctuary lined with shelves, a classic",
    "Lights flashed and music blared, a space alive with energy and movement, clearly a",
    "Rows of seats stretched out before the stage, the curtain ready to rise on tonight's performance at the",
    "Animals roamed freely, observed by families and children with delight, an afternoon spent at the",
    "The sound of waves crashing, sand between your toes, a serene day at the",
    "Stalls overflowed with fresh produce and handmade goods, a bustling morning at the",
    "Art adorned the walls, from classic to contemporary, visitors wandering through the",
    "Screens lit up with the latest blockbusters, a line forming at the popcorn stand, a night out at the",
    "The roar of the crowd, the thrill of the game, all eyes on the field at the",
    "Tents and stages set up, music echoing late into the night, the unmistakable vibe of a",
]


def gender_prompt(text, prompt):
    result = text.removeprefix(prompt).strip()
    if result in ["boy", "brother", "father", "man"]:
        color = "dark_blue"
    elif result in ["girl", "sister", "mother", "woman"]:
        color = "hot_pink3"
    else:
        color = "gray"
    return f"[{color}]{result}[/{color}]"


def wedding_prompt(text, prompt):
    result = text.removeprefix(prompt).strip()
    if result.lower() in ["wedding", "groom", "bride"]:
        color = "green"
    else:
        color = "gray"
    return f"[{color}]{result}[/{color}]"


def generate_text(prompts, coeff: float, func=gender_prompt):
    table = Table(
        "# L",
        *[f"Ex. {n}" for n in range(len(prompts))],
        title=f"One-word completion after adding vector with coeff {coeff} to different layers",
        show_lines=True,
        width=110,
    )

    with model.generate(
        max_new_tokens=3,
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
        "—", *[f"[b]{func(t, p)}[/b]" for p, t in zip(prompts, decoded_output)]
    )

    for layer in range(12):
        _, vector_dir = utils.current_version(dataset_dir / "vectors")
        sv = SteeringVector.load(vector_dir / f"layer_{layer}.pt", device="cpu")
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
            f"{layer}", *[func(t, p) for p, t in zip(prompts, decoded_output)]
        )

    rprint(table)


for coeff in [2]:
    generate_text(wedding_stubs[10:], coeff, func=wedding_prompt)


# %%
def generate_freeform(prompt, coeff, layer):
    sv = SteeringVector.load(
        dataset_dir / "vectors" / f"layer_{layer}.pt", device="cuda"
    )

    with model.generate(
        max_new_tokens=10,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.6,
        temperature=1,
    ) as runner:
        with runner.invoke(prompt, scan=False) as _:
            h = model.transformer.h[layer].output[0]
            h[:] += coeff * sv.vector
            steered_output = model.generator.output.save()
    print(model.tokenizer.batch_decode(steered_output)[0])


generate_freeform("Between a doll and a toy gun, I prefer the", 8, 8)


# %%
def has_keyword(text, keywords):
    if any(k in text for k in keywords):
        return any(re.findall(rf"\b{k}\b", text) for k in keywords)
    return False


def p_successful_rollout(
    model,
    prompt,
    coeffs=[0, 2, 4, 6, 8, 10, 12, 16, 20],
    length=40,
    batch_size=200,
    keywords=[
        "wedding",
        "weddings",
        "wed",
        "marry",
        "married",
        "marriage",
        "bride",
        "groom",
        "honeymoon",
    ],
):
    rollouts = []
    batch = [prompt] * batch_size

    with t.no_grad(), model.generate(
        batch,
        max_new_tokens=length,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.3,
        temperature=1,
    ) as _:
        baseline_output = model.generator.output.save()
    baseline_decoded = model.tokenizer.batch_decode(
        baseline_output, skip_special_tokens=True
    )
    baseline_p_success = (
        sum(any(k in d for k in keywords) for d in baseline_decoded) / batch_size
    )

    with Progress() as progress:
        task = progress.add_task(
            "Generating rollouts...", total=len(coeffs) * model.config.n_layer
        )
        for layer in [7, 8, 9, 10]:
            sv = SteeringVector.load(
                dataset_dir / "vectors" / f"layer_{layer}.pt", device="cuda"
            )
            for coeff in coeffs:
                with t.no_grad(), model.generate(
                    batch,
                    max_new_tokens=length,
                    scan=False,
                    validate=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1,
                ) as _:
                    model.transformer.h[layer].output[0][:] += coeff * sv.vector
                    steered_output = model.generator.output.save()
                decoded = model.tokenizer.batch_decode(
                    steered_output, skip_special_tokens=True
                )
                rollouts.append(
                    {
                        "layer": layer,
                        "p_success": sum(has_keyword(d, keywords) for d in decoded)
                        / batch_size,
                        "multiplier": coeff,
                    }
                )
                progress.update(task, advance=1)
    rollouts = pl.DataFrame(rollouts)
    fig = px.line(
        rollouts.to_pandas(),
        x="layer",
        y="p_success",
        color="coeff",
        markers=True,
        color_discrete_sequence=px.colors.sequential.OrRd,
    )
    fig.add_hline(y=baseline_p_success, line_dash="dot", line_color="red")
    return rollouts, fig


rollouts, fig = p_successful_rollout(model, "I think you're", batch_size=200)
fig.show()


# %%

layer = 10

baseline_perplexity = df.filter(c("layer").is_null())["loss"].median()
baseline_p_success = (
    rollouts.filter(c("multiplier") == 0)
    .filter(c("layer") == layer)["p_success"]
    .median()
)

pareto = (
    df.group_by("layer", "multiplier")
    .agg(c("loss").median().alias("mean_perplexity"))
    .join(rollouts, on=["layer", "multiplier"])
    .filter(c("layer") == layer)
    .sort("mean_perplexity", "p_success")
)

fig = px.line(
    pareto.to_pandas(),
    x="mean_perplexity",
    y="p_success",
    labels={"layer": "Layer"},
    markers=True,
    title=f"Pareto front for layer {layer}",
    custom_data=["multiplier"],
)
# fig.update_yaxes(matches=None)
# fig.update_xaxes(matches=None)
fig.add_hline(y=baseline_p_success, line_dash="dot", line_color="red")
fig.add_vline(x=baseline_perplexity, line_dash="dot", line_color="red")
fig.write_image(plot_dir / "pareto.pdf")
fig.update_traces(
    hovertemplate="<br>".join(["ColX: %{x}", "ColY: %{y}", "Col1: %{customdata[0]}"])
)

fig.show()
# %%
