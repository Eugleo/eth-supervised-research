from transformer_lens import HookedTransformer
import torch
from transformer_lens import utils, HookedTransformer
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

from sv.datasets import Dataset
from sae_analysis.analysis_utils import *

import torch
from tqdm import tqdm
import os


# load transformer lens model
model = HookedTransformer.from_pretrained_no_processing("gpt2-small", default_prepend_bos=False).eval()
model.to("cpu")

gpt2_small_sparse_autoencoders, gpt2_small_sae_sparsities = get_gpt2_res_jb_saes()
W_dec_all_layers = torch.stack(
    [
        gpt2_small_sparse_autoencoders[i].W_dec.detach().cpu()
        for i in gpt2_small_sparse_autoencoders.keys()
    ],
    dim=0,
)
print(gpt2_small_sparse_autoencoders)

# define what layer/module you want information from and get the internal activations
layer = 9
behavior = "gender"
max_features = 24576 
activations_path = f"data/{behavior}/deltas/{layer}_activations.pt"
activations = torch.load(activations_path)
deltas = torch.tensor(activations['delta'])
pos_activations = torch.tensor(activations['pos'])
neg_activations = torch.tensor(activations['neg'])
#Indices with the largest deltas

female_sv = torch.zeros(max_features)
female_features = [4148, 17423, 17709, 10162, 3466, 10822, 18751]
steering_vec = torch.zeros(max_features)
#steering_vec[female_features] =  1.5*neg_activations[female_features]
steering_vec[female_features] = 1.5*neg_activations[female_features]

    # reset the steering vector length to 1
    #print(steering_vec.norm())
hook_point = f"blocks.{layer}.hook_resid_pre"
sae = gpt2_small_sparse_autoencoders[hook_point]
steering_vec = steering_vec
steering_vec = sae.decode(steering_vec)
#steering_vec = sae.decode(steering_vec)
# reset the steering vector length to 1
#print(steering_vec.norm())
steering_vec /= steering_vec.norm()





# generate text while steering in positive direction

'''
# define the activation steering funtion
def act_add(steering_vec):
    def hook(activation, hook):
        return activation + steering_vec
    return hook


coefficients = [0.01, 0.05, 0.1, 0.5, 0.9, 1, 5, 10, 20]
responses = {}
for coeff in coefficients:
   

    print(f"Generating text with coeff: {coeff}")
    tag = f"coeff_{coeff}"
    cache_name = f"blocks.{layer}.hook_resid_post"
    model.add_hook(name=cache_name, hook=act_add(coeff*steering_vec))
    pos_response = model.generate(test_sentence, max_new_tokens=10, do_sample=True, top_k=50)
    model.reset_hooks()


    # generate text while steering in negative direction
    coeff = -coeff
    # reset the steering vector length to 1
    #print(steering_vec.norm())
    model.add_hook(name=cache_name, hook=act_add(coeff*steering_vec))
    neg_response = model.generate(test_sentence, max_new_tokens=10, do_sample=True, top_k=50)
    model.reset_hooks()
    
    responses[tag] = {"pos": pos_response, "neg": neg_response}

#Print Responses
for tag, response in responses.items():
    print(f"Tag: {tag}")
    print(f"Positive Response: {response['pos']}")
    print(f"Negative Response: {response['neg']}")
    print("\n")
'''