import sys
import os

import logging
import torch
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
        Can add more LLMs
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
        target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)
        target_embeddings = target_embeddings.detach().cpu().float()
        token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

        similarities = torch.matmul(target_embeddings, token_embeddings.T)
        top_k_values, top_k_indices = similarities.topk(k, dim=1)
        top_k_tokens = [self.tokenizer.convert_ids_to_tokens(indices) for indices in top_k_indices]

        return top_k_tokens, top_k_indices

    def visualize_prototype(self, prototype_embeddings):
        if isinstance(prototype_embeddings, str):
            prototype_embeddings = torch.load(prototype_embeddings)
            prototype_embeddings = torch.Tensor(prototype_embeddings).T
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
        if isinstance(prototype_embeddings, str):
            prototype_embeddings = torch.load(prototype_embeddings)
            prototype_embeddings = torch.Tensor(prototype_embeddings).T
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
                  "Text Prototypes", "Variables")

        top_tokens = []
        for target_idx in range(attention.shape[0]):
            top_scores_idx = np.argsort(attention[target_idx])[::-1][:10]
            top_token = [matched_tokens[idx] for idx in top_scores_idx]
            top_token_id = [matched_tokens_id[idx].tolist() for idx in top_scores_idx]
            top_tokens.append(top_token)
            logging.info(f"Variable Index: {target_idx}\nTop correlated tokens: {top_token}\n")
            logging.info(f"Variable Index: {target_idx}\nTop correlated tokens id: {top_token_id}\n")

        return top_tokens

    def visualize_aligned_textual_token_with_selected_words(self, aligned_textual_token, words):
        if isinstance(aligned_textual_token, str):
            aligned_textual_token = torch.load(aligned_textual_token)
        aligned_textual_token = aligned_textual_token[0].detach().cpu().float()
        aligned_textual_token = aligned_textual_token / aligned_textual_token.norm(dim=-1, keepdim=True)
        token_ids = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True).input_ids
        padding_id = self.tokenizer.pad_token_id

        similarities_list = []
        for idx, word_tokens in enumerate(token_ids):
            valid_token_ids = word_tokens[word_tokens != padding_id]
            valid_word_embeddings = self.vocab_embeddings[valid_token_ids].squeeze(1)
            valid_word_embeddings = valid_word_embeddings / valid_word_embeddings.norm(dim=-1, keepdim=True)

            # Calculate mean value if a word is represented by multiple tokens
            if valid_word_embeddings.shape[0] > 0:
                word_similarity = torch.matmul(aligned_textual_token, valid_word_embeddings.T)
                mean_similarity = torch.mean(word_similarity, dim=1, keepdim=True)
                similarities_list.append(mean_similarity)

        total_similarities = torch.cat(similarities_list, dim=1)
        top_k_indices = torch.topk(total_similarities, k=40, dim=1).indices

        for i in range(len(top_k_indices)):
            top_k_words = [words[idx] for idx in top_k_indices[i]]
            logging.info(f"Variables Index: {i}\nTok_k_tokens: {top_k_words}")

        self.plot(total_similarities.detach().numpy(),
                  "Aligned Textual Token with Selected Words",
                  "Selected Words", "Variables",
                  words, None, True)


def main():
    parser = argparse.ArgumentParser(description="Visualization for TS-Language Models")
    parser.add_argument("--function", type=str, required=True, default="prototype",
                        choices=["prototype", "attention_align", "aligned_tt_words"],
                        help="Choose function from "
                             "'prototype', 'attention_align', 'aligned_tt_words'")
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
    settings = 'No Specific!'
    if isinstance(args.embeddings_path, str):
        logging.info(f'\n*** {args.embeddings_path} ***\n')
        if "_e" in args.embeddings_path:
            settings = args.embeddings_path[args.embeddings_path.find("_e"):]
        else:
            settings = ""
    if isinstance(args.attention_path, str):
        logging.info(f'\n*** {args.attention_path} ***\n')
        settings = args.attention_path[args.attention_path.find("_e"):]

    visualizer = Visualizer(settings=settings, llm_name=args.llm_name, root_path=root_path)

    if args.function == "prototype":
        visualizer.visualize_prototype(args.embeddings_path)
    elif args.function == "attention_align":
        visualizer.visualize_attention_in_alignment(args.embeddings_path, args.attention_path)
    elif args.function == "aligned_tt_words":
        visualizer.visualize_aligned_textual_token_with_selected_words(args.embeddings_path, words)


if __name__ == "__main__":
    main()
