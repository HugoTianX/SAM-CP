

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

class BertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)


        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.gama = 0.10
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
        sent1 = input_ids[:,0,:]
        sent2 = input_ids[:,1,:]

        outputs1 = self.bert(sent1, attention_mask[:,0,:])
        outputs2 = self.bert(sent2, attention_mask[:,1,:])

        sequence_output1 = outputs1[0]
        sequence_output2 = outputs2[0]

        prediction_scores1 = self.cls(sequence_output1)
        prediction_scores2 = self.cls(sequence_output2)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss1 = loss_fct(prediction_scores1.view(-1, self.config.vocab_size), labels.view(-1))
        masked_lm_loss2 = loss_fct(prediction_scores2.view(-1, self.config.vocab_size), labels.view(-1))

        cos_sim = self.cos(sequence_output1.mean(1), sequence_output2.mean(1))

        loss = masked_lm_loss1 + masked_lm_loss2 + self.gama*cos_sim

        if not return_dict:
            output = (prediction_scores1,) + outputs1[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores1,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )
