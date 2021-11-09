from torch import nn, tanh, Tensor
import torch
from torchcrf import CRF

from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, init_logging

import logging
logger = logging.getLogger(__name__)


class NERModel(nn.Module):
    def __init__(self, vocab_size, lm=None, num_labels=2, hidden_dim=768):
        super(NERModel, self).__init__()
        self.lm = lm
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.bilstm = nn.LSTM(input_size=hidden_dim,
                              hidden_size=hidden_dim,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        self.clf_layer = nn.Linear(hidden_dim * 2, num_labels)
        if args.use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input, targets=None):
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        bilstm_out, _ = self.bilstm(hidden)
        cls_logits = self.clf_layer(tanh(bilstm_out))

        if args.use_crf:
            targets = targets[0]
            mask = torch.where(targets != torch.tensor(FILL_VAL).cuda(),
                               torch.ones(targets.shape, dtype=torch.uint8).cuda(),
                               torch.zeros(targets.shape, dtype=torch.uint8).cuda())
            output = -1 * self.crf(cls_logits, targets, mask=mask, reduction='token_mean')
        else:
            output = cls_logits

        return output

    def predict(self, input):
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        bilstm_out, _ = self.bilstm(hidden)
        cls_logits = self.clf_layer(tanh(bilstm_out))

        output = self.crf.decode(cls_logits)
        return output

    def lm_forward(self, input):
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        # print(hidden.shape)
        lm_logits = self.lm_head(hidden)
        return lm_logits
