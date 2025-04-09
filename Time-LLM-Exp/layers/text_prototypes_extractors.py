import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA


# *************************************
# *** Select text prototypes by PCA ***
# *************************************
class Prototypes_pca:
    def __init__(self, llm_model, num_prototypes):
        self.word_embeddings = llm_model.get_input_embeddings().weight
        d_llm = self.word_embeddings.shape[1]
        print('Prototypes PCA')
        num_prototypes = num_prototypes if num_prototypes < d_llm else 500
        self.pca = PCA(n_components=num_prototypes)
        self.prototypes_pca = self.pca.fit_transform(self.word_embeddings.T)
        print('Number of prototypes: {}'.format(self.prototypes_pca.shape[1]))

    def extract(self):
        return torch.tensor(self.prototypes_pca.T)

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')


# *****************************************
# *** Select text prototypes by K-means ***
# *****************************************
class Prototypes_kmeans:
    def __init__(self, llm_model, num_prototypes):
        print('Prototypes Kmeans')
        self.word_embeddings = llm_model.get_input_embeddings().weight
        self.num_prototypes = num_prototypes

        m, n = self.word_embeddings.shape
        result = torch.empty((m,), dtype=torch.int64, device=self.word_embeddings.device)
        indices = torch.randperm(m, device=self.word_embeddings.device)[:self.num_prototypes]
        self.prototypes_kmeans = self.word_embeddings[indices]
        count = 0
        max_iterations = 100
        embeddings_sq = torch.sum(self.word_embeddings ** 2, dim=1, keepdim=True)
        while count < max_iterations:
            prototypes_sq = torch.sum(self.prototypes_kmeans ** 2, dim=1)
            distances = embeddings_sq + prototypes_sq - 2 * torch.matmul(self.word_embeddings,
                                                                         self.prototypes_kmeans.t())
            distances = torch.sqrt(distances)

            index_min = torch.argmin(distances, dim=1)
            if torch.equal(index_min, result):
                break
            result = index_min
            for i in range(self.num_prototypes):
                items = self.word_embeddings[result == i]
                if items.shape[0] > 0:
                    self.prototypes_kmeans[i] = items.mean(dim=0)

            count += 1
        print('Num of prototypes:', self.prototypes_kmeans.shape[0])

    def extract(self):
        return self.prototypes_kmeans

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')


# ***************************************
# *** Select text prototypes randomly ***
# ***************************************
class Prototypes_random:
    def __init__(self, llm_model, num_prototypes):
        print('Prototypes Random')
        self.word_embeddings = llm_model.get_input_embeddings().weight
        self.num_prototypes = num_prototypes

        indices = torch.randperm(self.word_embeddings.shape[0])[:self.num_prototypes]
        self.prototypes_random = self.word_embeddings[indices]

    def extract(self):
        return self.prototypes_random

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')


# ****************************************************
# *** Select text prototypes by proposed texts ***
# ****************************************************
class Prototypes_text:
    def __init__(self, llm_model, tokenizer, texts, num_prototypes):
        print('Prototypes Text')
        self.word_embeddings = llm_model.get_input_embeddings().weight
        self.tokenizer = tokenizer
        self.texts = texts
        token_ids = self.tokenizer(self.texts, return_tensors='pt', padding=True, truncation=True).input_ids
        self.prototypes_text = self.word_embeddings[token_ids]
        d_llm = self.prototypes_text.shape[-1]
        self.prototypes_text = self.prototypes_text.reshape(-1, d_llm)
        if self.prototypes_text.shape[0] > num_prototypes:
            self.prototypes_text = self.prototypes_text[:num_prototypes]
        print('Num of text prototypes:', self.prototypes_text.shape[0])

    def extract(self):
        return self.prototypes_text

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')


# **********************************************
# *** Select text prototypes by linear layer ***
# **********************************************
class Prototypes_linear:
    def __init__(self, llm_model, num_prototypes):
        print('Prototypes Linear')
        self.word_embeddings = llm_model.get_input_embeddings().weight
        self.mapping_layer = nn.Linear(self.word_embeddings.shape[0], num_prototypes)
        print('Number of prototypes: {}'.format(num_prototypes))

    def extract(self):
        self.mapping_layer.to(torch.bfloat16)
        self.word_embeddings.to(torch.bfloat16)
        self.mapping_layer.to(self.word_embeddings.device)
        prototypes_linear = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        return prototypes_linear

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')


class Prototypes_similarity:
    def __init__(self, llm_model, tokenizer, sequences, num_prototypes):
        print('Prototypes Similarity')
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.num_prototypes = num_prototypes
        self.word_embeddings = llm_model.get_input_embeddings().weight

        # Tokenize the sequences and get embeddings for each token
        self.token_ids = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True).input_ids
        self.unique_ids = self.token_ids.unique()  # Get all unique token ids
        self.token_embeddings = self.word_embeddings[self.unique_ids]  # Get token embeddings of input sequences

        # Get sequences embeddings by average
        self.sequence_embeddings = []
        for ids in self.token_ids:
            self.sequence_embeddings.append(self.word_embeddings[ids].mean(dim=0))
        self.sequence_embeddings = torch.stack(self.sequence_embeddings)

        # Calculate similarities for individual tokens
        self.token_similarities = F.cosine_similarity(self.token_embeddings.unsqueeze(0),
                                                      self.word_embeddings.unsqueeze(1), dim=2)
        # Calculate similarities for sequence embeddings
        self.sequence_similarities = F.cosine_similarity(self.sequence_embeddings.unsqueeze(0),
                                                         self.word_embeddings.unsqueeze(1), dim=2)

        # Get top-k indices from both token and sequence similarities
        top_num = int(math.ceil(self.num_prototypes / (self.token_embeddings.shape[0] +
                                                       self.sequence_embeddings.shape[0])))
        token_topk_values, token_topk_indices = torch.topk(self.token_similarities, top_num, dim=0)
        sequence_topk_values, sequence_topk_indices = torch.topk(self.sequence_similarities, top_num, dim=0)

        print('top num:', top_num)

        # Combine and deduplicate indices
        total_indices = torch.cat([token_topk_indices.flatten(), sequence_topk_indices.flatten()])
        random_permutation = torch.randperm(total_indices.size(0))
        shuffled_total_indices = total_indices[random_permutation]

        # Select the unique prototypes
        self.prototypes_similarity = self.word_embeddings[shuffled_total_indices]
        if self.prototypes_similarity.shape[0] > num_prototypes:
            self.prototypes_similarity = self.word_embeddings[:num_prototypes]
        print('Number of prototypes:', self.prototypes_similarity.shape[0])

    def extract(self):
        return self.prototypes_similarity

    @staticmethod
    def save_prototypes(prototypes, filepath):
        torch.save(prototypes, filepath + '.pt')
