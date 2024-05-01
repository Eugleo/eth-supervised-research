# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import torch

# Data handling and utilities
from tqdm import tqdm
from datasets import load_dataset

# Local utility modules
from sae_vis.utils_fns import get_device
from sae_vis.model_fns import AutoEncoder
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.data_config_classes import SaeVisConfig

# Model and analysis tools from specific libraries
from transformer_lens import utils


class Layer_Deltas:
    
    cfg: SaeVisConfig
    encoder: AutoEncoder
    sae_data: SaeVisData
    deltas: list
    hook_point: str

    def __init__(self, pair_responses, model, encoder, hook_point, device, feature_size = 64, batch_size = 32):
        
        pos_tokens, neg_tokens = self.tokenize_behaviour_sets(pair_responses, model)
        #Move tokens to device
        pos_tokens = pos_tokens.to(device)
        neg_tokens = neg_tokens.to(device)

        model.to(device)

        cfg = self.construct_sae_vis_config(hook_point, feature_size, batch_size, verbose=False)


        sae_data_prior = self.construct_sae_vis_data(encoder, model, pos_tokens, cfg)

        sae_data_post = self.construct_sae_vis_data(encoder, model, neg_tokens, cfg)



        self.sae_data = sae_data_prior, sae_data_post

        self.deltas = self.construct_activation_deltas(sae_data_prior, sae_data_post)




    def tokenize_behaviour_sets(self, paired_responses, model, split = "train"):
        # Load the behaviour and non-behaviour sets
        
        pos_prompts = [response['pos'] for response in paired_responses]
        neg_prompts = [response['neg'] for response in paired_responses]

        # Tokenize the behaviour and non-behaviour sets
        tokenized_pos_data = model.tokenizer(pos_prompts, padding=True, truncation=True, return_tensors="pt")['input_ids']

        tokenized_neg_data = model.tokenizer(neg_prompts, padding=True, truncation=True, return_tensors="pt")['input_ids']

        # Get the tokens as a tensor
        all_tokens_pos = tokenized_pos_data
        all_tokens_neg = tokenized_neg_data

        assert isinstance(all_tokens_pos, torch.Tensor)
        assert isinstance(all_tokens_neg, torch.Tensor)
    
        return all_tokens_pos, all_tokens_neg

    def construct_sae_vis_config(self,hook_point, feature_size, batch_size, verbose=False):
        
        # Construct the SaeVisConfig object
        config = SaeVisConfig(
            hook_point=hook_point,
            features=range(feature_size),
            batch_size=batch_size,
            verbose=verbose
        )

        return config

    def construct_sae_vis_data(self,encoder, model, tokens, config):
        # Construct the SaeVisData object
        data = SaeVisData.create(
            encoder=encoder,
            model=model,
            tokens=tokens,
            cfg=config
        )

        return data

    def construct_activation_deltas(self,sae_vis_data_prior, sae_vis_data_post):

        feature_data_prior = sae_vis_data_prior.feature_data_dict
        feature_data_post = sae_vis_data_post.feature_data_dict
        assert len(feature_data_prior) == len(feature_data_post) #Same feature data lengths recorder

        num_features = len(feature_data_prior)

        max_activation_deltas = []
        max_activation_deltas_normalised = []


        for idx in range(num_features):
            prior_hist = feature_data_prior[idx].acts_histogram_data
            post_hist = feature_data_post[idx].acts_histogram_data

            try:
                max_prior = max(prior_hist.bar_values)
            except:
                max_prior = 0

            try:
                max_post = max(post_hist.bar_values)
            except:
                max_post = 0

            max_activation_delta = max_post - max_prior

            max_activation_deltas.append(max_activation_delta)


        return max_activation_deltas

    def plot_activation_deltas(self, save_path, show = True):
        deltas = self.deltas
        
        #Produce 2 plots, a scatter plot of each feature and the delta and the histogram of the deltas
        fig, axs = plt.subplots(1,2)
        fig.suptitle('Feature Activation Delta')
        #Set width and height of the plot
        fig.set_figheight(5)
        fig.set_figwidth(15)

        #Scatter plot and label the top 5 features
        axs[0].scatter(range(len(deltas)), deltas)
        axs[0].set_title('Feature Activation Delta')
        axs[0].set_xlabel('Feature Index')
        axs[0].set_ylabel('Activation Delta')
        #Label the top 5 features by absolute change
        top_5 = np.argsort(np.abs(deltas))[-5:]
        for i in top_5:
            axs[0].text(i, deltas[i], str(i), fontsize=12, color='red')

        #Histogram
        axs[1].hist(deltas, bins=20)
        axs[1].set_title('Feature Activation Delta Histogram')
        axs[1].set_xlabel('Activation Delta')
        axs[1].set_ylabel('Frequency')
        if show:
            plt.show()
        plt.savefig(save_path)

    def save_dashboard(self, layer_name, file_path):
        data_to_save_pos = self.sae_data[0]
        data_to_save_neg = self.sae_data[1]

        file_name_pos = f"feature_centric_vis_pos_{layer_name}.html"
        file_name_neg = f"feature_centric_vis_neg_{layer_name}.html"

        file_path_pos = file_path + "/" + file_name_pos
        file_path_neg = file_path + "/" + file_name_neg

        data_to_save_pos.save_feature_centric_vis(file_path_pos)
        data_to_save_neg.save_feature_centric_vis(file_path_neg)
        


class Feature_Deltas:
    layer_deltas: list[Layer_Deltas]

    def __init__(self, pair_responses, model, encoders, hook_points, device, feature_size = 64, batch_size = 32):
        self.layer_deltas = []
        pbar = tqdm(total = len(encoders))
        for layer, encoder, hook_point in zip(range(len(encoders)), encoders, hook_points):
            layer_deltas = Layer_Deltas(pair_responses, model, encoder, hook_point, device, feature_size, batch_size)
            self.layer_deltas.append(layer_deltas)
            pbar.update(1)
    