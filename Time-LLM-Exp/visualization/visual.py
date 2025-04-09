import sys
import os

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (GPT2Tokenizer, GPT2Model,
                          LlamaTokenizer, LlamaModel,
                          BertTokenizer, BertModel)
import seaborn as sns
import argparse
import re

sys.path.append(os.path.abspath('../'))
from utils.tsfel import *


class Visualizer:
    def __init__(self, settings, llm_name='gpt2', root_path=None):
        self.vocab_embeddings, self.tokenizer = self.get_wte(llm_name)
        self.settings_str = settings
        self.root_path = root_path

    @staticmethod
    def get_wte(llm_name):
        """
        Can add more LLMs (now support gpt2, llama, bert)
        """
        if llm_name == 'GPT2':
            tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
            model = GPT2Model.from_pretrained('openai-community/gpt2')
        elif llm_name == 'LLAMA':
            tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
            model = LlamaModel.from_pretrained('huggyllama/llama-7b')
        elif llm_name == 'BERT':
            tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
            model = BertModel.from_pretrained('google-bert/bert-base-uncased')
        else:
            raise ValueError("Unsupported LLM. Please choose from 'gpt2', 'llama', 'bert'.")
        word_embeddings = model.get_input_embeddings().weight

        # Add padding token
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token

        return word_embeddings, tokenizer

    def plot(self, matrix, title, xlabel='x', ylabel='y', xticklabels=None, yticklabels=None,
             rotate_xticks=False, rotate_yticks=False, figsize=(24, 12), cmap='viridis'):
        """
        Plot heatmap.
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(matrix, annot=False, fmt=".2f", cmap=cmap,
                    xticklabels=xticklabels if xticklabels is not None else False,
                    yticklabels=yticklabels if yticklabels is not None else False,
                    ax=ax)

        ax.set_aspect('equal')
        plt.title(title, fontsize=50)
        plt.xlabel(xlabel, fontsize=40)
        plt.ylabel(ylabel, fontsize=40)

        if rotate_xticks:
            plt.xticks(rotation=90)
        if rotate_yticks:
            plt.yticks(rotation=90)
        plt.tight_layout()
        filename = f'./{self.root_path}/{title.replace(" ", "_").lower()}{self.settings_str}.png'
        plt.savefig(filename)
        plt.show()

        return filename

    def match_tokens(self, target_embeddings, token_embeddings, k=3):
        """
        Given target_embeddings and token_embeddings, output the top k most matched tokens
        for each target_embedding.

        :param target_embeddings: Tensor of target embeddings
        :param token_embeddings: Tensor of token embeddings
        :param k: Number of top tokens to retrieve
        :return: List of lists containing the top k matched tokens for each target embedding
        """
        target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)
        target_embeddings = target_embeddings.detach().cpu().float()
        token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

        similarities = torch.matmul(target_embeddings, token_embeddings.T)
        top_k_values, top_k_indices = similarities.topk(k, dim=1)
        top_k_tokens = [self.tokenizer.convert_ids_to_tokens(indices) for indices in top_k_indices]

        return top_k_tokens, top_k_indices

    def visualize_prototype(self, prototype_embeddings):
        """
        Given prototype_embeddings, output the most matched token (that is semantic representation) of each prototype
        and the similarity matrix of prototypes.

        :param prototype_embeddings: Tensor of prototype embeddings
        :return: List of lists containing the top k most matched tokens for each prototype
        """
        if isinstance(prototype_embeddings, str):
            prototype_embeddings = torch.load(prototype_embeddings)

        matched_tokens, matched_tokens_id = self.match_tokens(prototype_embeddings, self.vocab_embeddings, k=1)
        logging.info(f'The tokens corresponding to the prototype embeddings:\n {matched_tokens}')
        logging.info(f'The tokens id corresponding to the prototype embeddings:\n {matched_tokens_id.tolist()}')

        x = prototype_embeddings.detach().cpu().float()
        norms = torch.norm(x, dim=1)
        norms = torch.where(norms == torch.tensor(0), torch.tensor(1.0), norms)
        x_normalized = x / norms[:, None]
        similarity = torch.mm(x_normalized, x_normalized.T)

        self.plot(similarity, "Similarity Matrix of Prototypes",
                  "Prototype Index", "Prototype Index",
                  figsize=(14, 14), cmap='coolwarm')

        return matched_tokens

    def visualize_attention_in_alignment(self, prototype_embeddings, attention_weights):
        """
        Given prototype_embeddings and attention_weights, output the attention matrix heatmap and
        top k correlated prototypes of each patch.

        :param prototype_embeddings: Tensor of prototype embeddings
        :param attention_weights: Tensor of attention weights
        :return: List of lists containing the top k most correlated prototypes for each patch
        """
        if isinstance(prototype_embeddings, str):
            prototype_embeddings = torch.load(prototype_embeddings)
        if isinstance(attention_weights, str):
            attention_weights = torch.load(attention_weights)

        matched_tokens, matched_tokens_id = self.match_tokens(prototype_embeddings, self.vocab_embeddings, k=1)
        if attention_weights.dim() == 4:
            attention = attention_weights[0].float().mean(dim=0).detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attention = attention_weights[0].float().detach().cpu().numpy()
        else:
            attention = attention_weights.float().detach().cpu().numpy()
        self.plot(attention, "Attention Matrix in Alignment Module",
                  "Text Prototypes", "Patches")

        top_tokens = []
        for target_idx in range(attention.shape[0]):
            top_scores_idx = np.argsort(attention[target_idx])[::-1][:10]
            top_token = [matched_tokens[idx] for idx in top_scores_idx]
            top_token_id = [matched_tokens_id[idx].tolist() for idx in top_scores_idx]
            top_tokens.append(top_token)
            logging.info(f"Patches Index: {target_idx}\nTop correlated tokens: {top_token}\n")
            logging.info(f"Patches Index: {target_idx}\nTop correlated tokens id: {top_token_id}\n")

        return top_tokens

    def visualize_attention_in_llm(self, epoch, attention_weights, prompts_id_path):
        """
        Given attention_weights from LLM and corresponding prompts, output the attention matrix heatmap.

        :param epoch: Given epoch checkpoint (If needed)
        :param attention_weights: Tensor of attention weights
        :param prompts_id_path: Path to the prompts id
        """
        epoch = epoch
        if isinstance(attention_weights, str):
            epoch = int(re.search(r'e(\d+)', attention_weights).group(1))
            attention_weights = torch.load(attention_weights)
        if attention_weights.dim() == 5:
            attention = attention_weights[-1][0].float().mean(dim=0).detach().cpu().numpy()
        elif attention_weights.dim() == 4:
            attention = attention_weights[0].float().mean(dim=0).detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attention = attention_weights[0].float().detach().cpu().numpy()
        else:
            attention = attention_weights.float().detach().cpu().numpy()

        with open(prompts_id_path, 'r') as file:
            prompts_info = file.read()
        epoch_point = f"******Epoch_{epoch}******"
        start_index = prompts_info.index(epoch_point)
        next_epoch_point = f"******Epoch_{epoch + 10}******"
        if next_epoch_point in prompts_info:
            end_index = prompts_info.index(next_epoch_point, start_index)
        else:
            end_index = len(prompts_info)
        prompts_id = prompts_info[start_index:end_index].strip()
        prompts_id = [int(num) for num in prompts_id.split()[1:]]

        attention_len = attention.shape[-1]
        prompts_token = self.tokenizer.convert_ids_to_tokens(prompts_id)
        logging.info(f'Prompts: \n{prompts_token}')
        prompts_length = len(prompts_token)
        ticklabels = []
        ticklabels.extend(prompts_token)
        for i in range(attention_len - prompts_length):
            ticklabels.append(f"p-{i}")

        top_k_tokens = {}

        logging.info('Tokens cross relation from attention in LLMs:\n')
        for i in range(len(ticklabels)):
            attention_weights = attention[i]
            top_k_idx = np.argsort(attention_weights)[-10:][::-1]
            top_k_tokens[ticklabels[i]] = [ticklabels[idx] for idx in top_k_idx]
            logging.info(f'For token {ticklabels[i]}, top 5 highest weights tokens to it:\n {top_k_tokens[ticklabels[i]]}')

        self.plot(attention, "Attention Matrix in LLM",
                  "Text Prototypes", "Patches",
                  ticklabels, ticklabels, figsize=(28, 24),
                  rotate_xticks=True)

    def visualize_aligned_ts_emb(self, aligned_ts_emb, k=1):
        """
        Given aligned_ts_emb, output the most matched token (that is semantic representation) of each patch.

        :param aligned_ts_emb: Tensor of aligned time series embeddings
        :return: List of lists containing the top k most matched tokens for each patch
        """
        if isinstance(aligned_ts_emb, str):
            aligned_ts_emb = torch.load(aligned_ts_emb)
        aligned_ts_emb = aligned_ts_emb[0]
        top_k_tokens, top_k_tokens_id = self.match_tokens(aligned_ts_emb, self.vocab_embeddings, k=k)
        logging.info(f'The tokens corresponding to aligned time series embeddings:\n {top_k_tokens}')
        logging.info(f'The tokens id corresponding to aligned time series embeddings:\n {top_k_tokens_id.tolist()}')

        x = aligned_ts_emb.detach().cpu().float()
        norms = torch.norm(x, dim=1)
        norms = torch.where(norms == torch.tensor(0), torch.tensor(1.0), norms)
        x_normalized = x / norms[:, None]
        similarity = torch.mm(x_normalized, x_normalized.T)
        self.plot(similarity, "Similarity Matrix of Patches",
                  "Patch Index", "Patch Index",
                  figsize=(14, 14), cmap='coolwarm')

        return top_k_tokens

    def visualize_aligned_ts_emb_with_selected_words(self, aligned_ts_emb, words):
        """
        Given aligned_ts_emb and selected_words, output the similarity matrix between each patch and word,
        and output top k correlated words of each patch.

        :param aligned_ts_emb: Tensor of aligned time series embeddings
        :param words: List of selected words
        """
        if isinstance(aligned_ts_emb, str):
            aligned_ts_emb = torch.load(aligned_ts_emb)
        aligned_ts_emb = aligned_ts_emb[0].detach().cpu().float()
        aligned_ts_emb = aligned_ts_emb / aligned_ts_emb.norm(dim=-1, keepdim=True)
        token_ids = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True).input_ids
        padding_id = self.tokenizer.pad_token_id

        similarities_list = []
        for idx, word_tokens in enumerate(token_ids):
            valid_token_ids = word_tokens[word_tokens != padding_id]
            valid_word_embeddings = self.vocab_embeddings[valid_token_ids].squeeze(1)
            valid_word_embeddings = valid_word_embeddings / valid_word_embeddings.norm(dim=-1, keepdim=True)

            # Calculate mean value if a word is represented by multiple tokens
            if valid_word_embeddings.shape[0] > 0:
                word_similarity = torch.matmul(aligned_ts_emb, valid_word_embeddings.T)
                mean_similarity = torch.mean(word_similarity, dim=1, keepdim=True)
                similarities_list.append(mean_similarity)

        total_similarities = torch.cat(similarities_list, dim=1)
        top_k_indices = torch.topk(total_similarities, k=40, dim=1).indices

        for i in range(len(top_k_indices)):
            top_k_words = [words[idx] for idx in top_k_indices[i]]
            logging.info(f"Patches Index: {i}\nTok_k_words: {top_k_words}")

        self.plot(total_similarities.detach().numpy(),
                  "Aligned Time Series Embeddings with Selected Words",
                  "Selected Words", "Patches",
                  words, None, True)

    def visualize_prototype_emb_with_selected_words(self, prototype_embeddings, words):
        """
        Given prototype_embeddings and selected_words, output the similarity matrix between each prototype and word,
        and output top k correlated words of each prototype.

        :param prototype_embeddings: Tensor of prototype embeddings
        :param words: List of selected words
        """
        if isinstance(prototype_embeddings, str):
            prototype_embeddings = torch.load(prototype_embeddings)
        prototype_embeddings = prototype_embeddings.detach().cpu().float()
        prototype_embeddings = prototype_embeddings / prototype_embeddings.norm(dim=-1, keepdim=True)
        token_ids = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True).input_ids
        padding_id = self.tokenizer.pad_token_id

        similarities_list = []
        for idx, word_tokens in enumerate(token_ids):
            valid_token_ids = word_tokens[word_tokens != padding_id]
            valid_embeddings = self.vocab_embeddings[valid_token_ids].squeeze(1)
            valid_embeddings = valid_embeddings / valid_embeddings.norm(dim=-1, keepdim=True)

            # Calculate mean value if a word is represented by multiple tokens
            if valid_embeddings.shape[0] > 0:
                word_similarity = torch.matmul(prototype_embeddings, valid_embeddings.T)
                mean_similarity = torch.mean(word_similarity, dim=1, keepdim=True)
                similarities_list.append(mean_similarity)

        total_similarities = torch.cat(similarities_list, dim=1)

        top_k_indices = torch.topk(total_similarities, k=40, dim=1).indices

        for i in range(len(top_k_indices)):
            top_k_words = [words[idx] for idx in top_k_indices[i]]
            logging.info(f"Prototype Index: {i}\nTop_k_words: {top_k_words}")

        self.plot(total_similarities.detach().numpy(),
                  "Prototype Embeddings with Selected Words",
                  "Selected Words", "Text Prototypes",
                  words, None, True)

    def visualize_patches_with_semantic_ts(self, epoch, patches_path, aligned_ts_emb):
        """
        Given scaled patches and aligned_ts_emb, output the most matched token (that is semantic representation)
        of each patch and patches line chart of each token.

        :param epoch: Given epoch checkpoint (If needed)
        :param patches_path: Path to patches file
        :param aligned_ts_emb: Tensor of aligned time series embeddings
        """
        epoch = epoch
        if isinstance(aligned_ts_emb, str):
            epoch = int(re.search(r'e(\d+)', aligned_ts_emb).group(1))
            aligned_ts_emb = torch.load(aligned_ts_emb)

        with open(patches_path, 'r') as file:
            patches_info = file.read()
        epoch_point = f"******Epoch_{epoch}******"
        start_index = patches_info.index(epoch_point)
        next_epoch_point = f"******Epoch_{epoch + 5}******"
        if next_epoch_point in patches_info:
            end_index = patches_info.index(next_epoch_point, start_index)
        else:
            end_index = len(patches_info)
        scaled_patches = patches_info[start_index:end_index].strip()
        scaled_patches_lines = scaled_patches.split('\n')[1:]

        token_patch_dict = {}
        tokens = self.visualize_aligned_ts_emb(aligned_ts_emb, k=20)
        for token, scaled_patches_line in zip(tokens, scaled_patches_lines):
            token = tuple(token)
            patch = np.array(list(map(float, scaled_patches_line.split())))
            if token in token_patch_dict:
                token_patch_dict[token].append(patch)
            else:
                token_patch_dict[token] = [patch]

        semantic_matching_index, feature_diff_metrics = self.cal_semantic_matching_index(token_patch_dict)
        logging.info(f'Semantic Matching Index: {semantic_matching_index}')

        for token_set_idx, (token, patches) in enumerate(token_patch_dict.items()):
            plt.figure(figsize=(28, 14))
            patches_stack = np.stack(patches)
            patch_mean = np.mean(patches_stack, axis=0)
            patch_std = np.std(patches_stack, axis=0)

            total_mean = np.mean(patch_mean)
            total_std = np.mean(patch_std)
            total_max = np.max(patches_stack)
            total_min = np.min(patches_stack)
            total_median = np.median(patches_stack)

            for i, patch in enumerate(patches):
                plt.plot(patch, alpha=0.5, linewidth=5)

            plt.plot(patch_mean, color='black', linewidth=8)
            plt.fill_between(range(len(patch_mean)), patch_mean - patch_std, patch_mean + patch_std, color='blue',
                             alpha=0.1)

            stat_info = '\n'.join((
                f'Patches Number : {len(patches)}',
                f'Mean: {total_mean:.2f}',
                f'Std: {total_std:.2f}',
                f'Max: {total_max:.2f}',
                f'Min: {total_min:.2f}',
                f'Median: {total_median:.2f}',
            ))
            props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
            plt.gca().text(1.02, 0.5, stat_info, transform=plt.gca().transAxes, fontsize=30,
                           verticalalignment='center', bbox=props, fontweight='bold')

            plt.title(f"{token} - Patches Visualization", fontsize=50)
            plt.xlabel("Step", fontsize=40)
            plt.ylabel("Value", fontsize=40)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.subplots_adjust(right=0.8)
            token_sets_path = f"./{self.root_path}/token_sets"
            if not os.path.exists(token_sets_path):
                os.makedirs(token_sets_path)
            plt.savefig(f"./{token_sets_path}/token_set_{token_set_idx}{self.settings_str}.png")
            plt.close()

    @staticmethod
    def cal_semantic_matching_index(token_patch_dict):
        """
        Calculate Semantic Matching Index (SMI)
        """
        a = 0.5
        b = 0.1
        
        feature_diff_metrics = {}
        feature_extraction_funcs = feature_extraction_functions

        for token_set, patches in token_patch_dict.items():
            feature_values = {func.__name__: [func(patch) for patch in patches] for func in feature_extraction_funcs}
            token_set_features = {}
            
            for func in feature_extraction_funcs:
                feature_name = func.__name__
                token_set_features[f'mean_{feature_name}'] = np.mean(feature_values[feature_name])
                token_set_features[f'std_{feature_name}'] = np.std(feature_values[feature_name])

            feature_diff_metrics[token_set] = token_set_features

        token_sets = list(token_patch_dict.keys())
        num_token_sets = len(token_sets)

        diff_within_token_set = sum(
            sum(feature_diff_metrics[token_set][f'std_{func.__name__}'] for func in feature_extraction_funcs)
            for token_set in token_sets
        )
        avg_diff_within = diff_within_token_set / num_token_sets

        diff_between_token_set = 0
        comp_count = 0
        for i in range(num_token_sets - 1):
            for j in range(i + 1, num_token_sets):
                diff_sum = sum(
                    abs(feature_diff_metrics[token_sets[i]][f'mean_{func.__name__}'] -
                        feature_diff_metrics[token_sets[j]][f'mean_{func.__name__}'])
                    for func in feature_extraction_funcs
                )
                diff_between_token_set += diff_sum
                comp_count += 1

        # Calculate average difference between token sets
        avg_diff_between = diff_between_token_set / comp_count if comp_count > 0 else 0

        if avg_diff_within < 1e-8:
            semantic_matching_ratio = float('inf')
        else:
            semantic_matching_ratio = a * avg_diff_between / avg_diff_within
        
        semantic_matching_index = 1 - np.exp(-b * semantic_matching_ratio)

        return semantic_matching_index, feature_diff_metrics


def main():
    parser = argparse.ArgumentParser(description="Visualization for TS-Language Models")
    parser.add_argument("--function", type=str, required=True, default="prototype",
                        choices=["prototype", "attention_align", "attention_llm", "aligned_ts",
                                 "aligned_ts_words", "prototype_words", "patches_semantic_ts"],
                        help="Choose function from "
                             "'prototype', 'attention_align', 'attention_llm', 'aligned_ts', "
                             "'aligned_ts_words', 'prototype_words', 'patches_semantic_ts'")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to the embeddings file")
    parser.add_argument("--attention_path", type=str, default=None,
                        help="Path to the attention file")
    parser.add_argument("--selected_words_path", type=str, default=None,
                        help="Path to the selected words file")
    parser.add_argument("--log_file_name", type=str, default="./results.log",
                        help="Path to save the log file")
    parser.add_argument("--llm_name", type=str, default="gpt2",
                        help="Name of the language model (e.g., gpt2, llama, bert)")
    parser.add_argument("--patches_path", type=str, default=None,
                        help="Path to the patches file")
    parser.add_argument("--prompts_id_path", type=str, default=None,
                        help="Path to the prompts id file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to the output files")

    args = parser.parse_args()
    if args.selected_words_path:
        with open(args.selected_words_path, "r") as file:
            words = [line.strip().strip('"') for line in file.readlines()]

    root_path = f'./visualization_results/{args.output_path}'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    logging.basicConfig(filename=f'{root_path}/{args.log_file_name}', level=logging.INFO, format='%(message)s')
    settings = 'No Specific'
    if isinstance(args.embeddings_path, str):
        logging.info(f'\n*** {args.embeddings_path} ***\n')
        settings = args.embeddings_path[args.embeddings_path.find("_e"):]
    if isinstance(args.attention_path, str):
        logging.info(f'\n*** {args.attention_path} ***\n')
        settings = args.attention_path[args.attention_path.find("_e"):]

    visualizer = Visualizer(settings=settings, llm_name=args.llm_name, root_path=root_path)

    if args.function == "prototype":
        visualizer.visualize_prototype(args.embeddings_path)
    elif args.function == "attention_align":
        visualizer.visualize_attention_in_alignment(args.embeddings_path, args.attention_path)
    elif args.function == "attention_llm":
        visualizer.visualize_attention_in_llm(10, args.attention_path, args.prompts_id_path)
    elif args.function == "aligned_ts":
        visualizer.visualize_aligned_ts_emb(args.embeddings_path)
    elif args.function == "aligned_ts_words":
        visualizer.visualize_aligned_ts_emb_with_selected_words(args.embeddings_path, words)
    elif args.function == "prototype_words":
        visualizer.visualize_prototype_emb_with_selected_words(args.embeddings_path, words)
    elif args.function == "patches_semantic_ts":
        visualizer.visualize_patches_with_semantic_ts(5, args.patches_path, args.embeddings_path)


if __name__ == "__main__":
    main()
