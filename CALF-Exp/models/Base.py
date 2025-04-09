import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from models.GPT2_arch import AccustumGPT2Model
from transformers import AutoTokenizer

from .Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )

        self.task_name = configs.task_name

        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                           output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        self.linear = nn.Linear(configs.seq_len, configs.d_model)

        for layer in (self.gpt2, self.out_layer, self.time_proj, self.linear):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forecast(self, x):
        B, L, M = x.shape  # (batch_size, length, num_channels)
        # print('time series shape (origin):', x.shape)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')  # (batch_size, num_channels, length)
        outputs_time1 = self.linear(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        # residue connection
        outputs_time += outputs_time1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])

        outputs_time = self.out_layer(outputs_time[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
        }

    # def classification(self, x):
    #     B, L, M = x.shape
    #
    #     x = rearrange(x, 'b l m -> b m l')
    #
    #     outputs_time1, outputs_text1 = self.in_layer(x)
    #
    #     outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
    #     outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
    #
    #     outputs_time += outputs_time1
    #     outputs_text += outputs_text1
    #
    #     intermidiate_feat_time = tuple(
    #         [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
    #     intermidiate_feat_text = tuple(
    #         [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
    #
    #     outputs_time = outputs_time.reshape(B, -1)
    #     outputs_text = outputs_text.reshape(B, -1)
    #
    #     outputs_time = self.out_layer(outputs_time)
    #     outputs_text = self.out_layer(outputs_text)
    #
    #     return {
    #         'outputs_text': outputs_text,
    #         'outputs_time': outputs_time,
    #         'intermidiate_time': intermidiate_feat_time,
    #         'intermidiate_text': intermidiate_feat_text,
    #     }
    #
    # def imputation(self, x, mask):
    #     B, L, M = x.shape
    #
    #     means = x.mean(1, keepdim=True).detach()
    #     x = x - means
    #     x = x.masked_fill(mask == 0, 0)
    #
    #     stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
    #     x /= stdev
    #
    #     x = rearrange(x, 'b l m -> b m l')
    #
    #     outputs_time1, outputs_text1 = self.in_layer(x)
    #
    #     outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
    #     outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
    #     # residue connection
    #     outputs_time += outputs_time1
    #     outputs_text += outputs_text1
    #
    #     intermidiate_feat_time = tuple(
    #         [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
    #     intermidiate_feat_text = tuple(
    #         [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
    #
    #     outputs_time = self.out_layer(outputs_time)
    #     outputs_text = self.out_layer(outputs_text)
    #
    #     outputs_time = rearrange(outputs_time, 'b m l -> b l m')
    #     outputs_text = rearrange(outputs_text, 'b m l -> b l m')
    #
    #     outputs_text = outputs_text * stdev + means
    #     outputs_time = outputs_time * stdev + means
    #
    #     return {
    #         'outputs_text': outputs_text,
    #         'outputs_time': outputs_time,
    #         'intermidiate_time': intermidiate_feat_time,
    #         'intermidiate_text': intermidiate_feat_text,
    #     }
    #
    # def anomaly_detection(self, x):
    #     B, L, M = x.shape
    #
    #     means = x.mean(1, keepdim=True).detach()
    #     x = x - means
    #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    #     x /= stdev
    #
    #     x = rearrange(x, 'b l m -> b m l')
    #
    #     outputs_time1, outputs_text1 = self.in_layer(x)
    #
    #     outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
    #     outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
    #     # residue connection
    #     outputs_time += outputs_time1
    #     outputs_text += outputs_text1
    #
    #     intermidiate_feat_time = tuple(
    #         [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
    #     intermidiate_feat_text = tuple(
    #         [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
    #
    #     outputs_time = self.out_layer(outputs_time)
    #     outputs_text = self.out_layer(outputs_text)
    #
    #     outputs_time = rearrange(outputs_time, 'b m l -> b l m')
    #     outputs_text = rearrange(outputs_text, 'b m l -> b l m')
    #
    #     outputs_text = outputs_text * stdev + means
    #     outputs_time = outputs_time * stdev + means
    #
    #     return {
    #         'outputs_text': outputs_text,
    #         'outputs_time': outputs_time,
    #         'intermidiate_time': intermidiate_feat_time,
    #         'intermidiate_text': intermidiate_feat_text,
    #     }

    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
