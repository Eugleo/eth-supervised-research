from transformer_lens import utils, HookedTransformer
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

from sv.datasets import Dataset
from sae_analysis.analysis_utils import *

import torch

model = HookedTransformer.from_pretrained("gpt2-small")
gpt2_small_sparse_autoencoders, gpt2_small_sae_sparsities = get_gpt2_res_jb_saes()
W_dec_all_layers = torch.stack(
    [
        gpt2_small_sparse_autoencoders[i].W_dec.detach().cpu()
        for i in gpt2_small_sparse_autoencoders.keys()
    ],
    dim=0,
)

#Load response data
data_path = "data/gender/dataset/v1/generate.json"
dataset = Dataset.load(data_path)
pairs = dataset.as_prefixed_pairs()

#Cycle through each layer and construct sae visualisation data and deltas
for layer in tqdm(range(12)):
    hook_point = f"blocks.{layer}.hook_resid_pre"
    encoder = gpt2_small_sparse_autoencoders[hook_point].to("cpu")

    layer_deltas = Layer_Deltas(pairs, model, encoder, hook_point, device="cpu")
    
    plot_path = "data/gender/plots/feature_deltas/feature_deltas_layer_" + str(layer) + ".png"
    dashboard_path = "data/gender/sae_visualisation"

    layer_deltas.plot_activation_deltas(plot_path, show=False)
    layer_deltas.save_dashboard(layer, dashboard_path)
   
