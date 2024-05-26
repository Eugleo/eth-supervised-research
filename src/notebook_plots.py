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

device = "cpu"

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
model.to(device)

DATASET = "gender"

dataset_dir = Path("../data") / DATASET
scores_version, scores_dir = utils.get_version(dataset_dir / "scores", version=1)
data = (
    pl.concat([pl.read_csv(file) for file in scores_dir.glob("*.csv")])
    .filter(c("loss").is_not_nan())
    .sort("layer", "multiplier")
)

plots_version, plots_dir = utils.next_version(dataset_dir / "plots")
# %%
fig = plots.loss_change_per_layer(
    data,
    f"Change of loss after intervention (lower is better), scores v{scores_version}",
)
fig.write_image(plots_dir / "loss_change_per_layer.pdf")
fig.show()

# %%
fig = plots.loss_difference_per_layer(
    data,
    title=f"The effectivenes of steering (higher is better), scores v{scores_version}",
)
fig.write_image(plots_dir / "loss_difference_per_layer.pdf")
fig.show()

# %%
gender_stubs = [
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
    if any(w in result for w in ["boy", "brother", "father", "man"]):
        color = "dark_blue"
    elif any(w in result for w in ["girl", "sister", "mother", "woman"]):
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
        "Prompt",
        "No steering",
        f"Steering with {coeff} × vector",
        title="One-word completion for a prompt",
        show_lines=True,
        width=80,
    )

    with model.generate(
        max_new_tokens=3,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
    ) as runner:
        with runner.invoke(prompts, scan=False) as _:
            default_output = model.generator.output.save()

    default_output = model.tokenizer.batch_decode(
        default_output, skip_special_tokens=True
    )

    for layer in [8]:
        sv = SteeringVector.load(
            Path("../data/gender/vectors/v1") / f"layer_8.pt", device=device
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
        for prompt, unsteered, steered in zip(prompts, default_output, decoded_output):
            table.add_row(prompt, func(unsteered, prompt), func(steered, prompt))

    rprint(table)


for coeff in [1]:
    generate_text(gender_stubs[:3], -coeff, func=gender_prompt)


# %%
_, baseline_vector_dir = utils.get_version(dataset_dir / "vectors", version=1)
baseline_vector = SteeringVector.load(baseline_vector_dir / "layer_8.pt", device="cuda")

_, vector_dir = utils.get_version(dataset_dir / "vectors", version=8)
vector = SteeringVector.load(vector_dir / "layer_8.pt", device="cuda")


sv = baseline_vector * 3
sv = vector * 1.5


def generate_freeform(start_prompt, vector):
    completions = Table(
        "Completion",
        show_lines=True,
        width=60,
    )
    for _ in range(5):
        with t.no_grad(), model.generate(
            start_prompt,
            max_new_tokens=40,
            scan=False,
            validate=False,
            pad_token_id=model.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.6,
            temperature=1,
        ) as _:
            model.transformer.h[vector.layer].output[0][:] += vector.vector
            steered_output = model.generator.output.save()
        prompt = model.tokenizer.batch_decode(steered_output.value)[0]
        completions.add_row(prompt)
    rprint(completions)


# sv = SteeringVector.load(vector_dir / "layer_8.pt", device="cuda") * 3
generate_freeform("My favorite doll is a little", sv)


# %%
from sv import scoring

GENDER_KEYWORDS = [
    "dad",
    "father",
    "brother",
    "man",
    "men",
    "dude",
    "buddy",
    "boy",
    "guy",
    "gun",
    "prince",
    "king",
    "husband",
]

vector = SteeringVector.load(
    Path("../data/gender/vectors/v1") / f"layer_8.pt", device=device
)

result = scoring.p_successful_rollout(
    model,
    "My favorite doll is a little",
    vector,
    GENDER_KEYWORDS,
    multipliers=[3],
)

# %%

results = []
for layer in range(12):
    vector = SteeringVector.load(
        Path("../data/gender/vectors/v1") / f"layer_{layer}.pt", device=device
    )
    p_success = scoring.p_successful_rollout(
        model,
        "My favorite doll is a little",
        vector,
        GENDER_KEYWORDS,
        multipliers=[3],
    )
    results += [
        {"layer": layer, "p_success": p_success[3], "type": "steered"},
        {"layer": layer, "p_success": p_success[0], "type": "unsteered"},
    ]

# %%

df = pl.DataFrame(results)

fig = px.line(
    df.filter(c("type") == "steered"),
    x="layer",
    y="p_success",
    color="type",
    title="Probability of Masculine Keywords in Completion",
    labels={
        "layer": "Layer with Applied Steering Vector",
        "p_success": "P(masculine keyword present)",
        "type": "Model",
    },
    color_discrete_map={"steered": "steelblue", "unsteered": "silver"},
)

fig.update_layout(showlegend=False)

fig.add_hline(
    y=df.filter(c("type") == "unsteered").get_column("p_success").mean(),
    line_dash="dash",
    annotation_text="Unsteered baseline",
)

fig.show()
