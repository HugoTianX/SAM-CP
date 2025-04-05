

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,

)
from transformers import BertConfig, BertPreTrainedModel, BertModel, BertOnlyMLMHead


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)


        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.mlp = MLPLayer(config.hidden_size, config.hidden_size)
        self.sim = Similarity(temp=0.05)
        self.delta = 0.05
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) :
        
        batch_size = input_ids.size(0)
        num_sent = 2
        input_ids = input_ids.view((-1, input_ids.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs[0] 
        prediction_scores = self.cls(sequence_output) #
        loss_fct = CrossEntropyLoss()
        masked_lm_loss1 = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = cls_output.view((batch_size, num_sent, cls_output.size(-1)))
        cls_output = self.mlp(cls_output)
        z1, z2 = cls_output[:, 0], cls_output[:, 1]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels_cse = torch.arange(cos_sim.size(0)).long().cuda()
        loss_cse = loss_fct(cos_sim, labels_cse)

        loss = masked_lm_loss1 + self.delta*loss_cse

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
