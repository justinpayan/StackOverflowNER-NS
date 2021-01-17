from torch import nn, tanh, Tensor
import torch
from torchcrf import CRF

from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, init_logging

import logging
logger = logging.getLogger(__name__)

# could go get BiLSTM and CRF from https://github.com/lonePatient/BiLSTM-CRF-NER-PyTorch,
# if performance seems like it'll be better with that.


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
        # logging.info("input in fwd: %s" % str(input))
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        bilstm_out, _ = self.bilstm(hidden)
        cls_logits = self.clf_layer(tanh(bilstm_out))
        # if torch.isnan(cls_logits).any():
            # for i in range(input.shape[0]):
            #     logger.info("decoded input: %s" % str(TOKENIZER.decode(input[i, :])))

        # logger.info("hidden shape %s" % str(hidden.shape))
        # logger.info("bilstm_out.shape: %s" % str(bilstm_out.shape))
        # logger.info("tanh(bilstm_out): %s" % str(tanh(bilstm_out)))
        # logger.info("cls_logits: %s" % str(cls_logits))

        if args.use_crf:
            # logger.info("targets len: %d" % len(targets))
            # logger.info("targets[0] shape: %s" % str(targets[0].shape))
            # logger.info("targets[1] shape: %s" % str(targets[1].shape))
            targets = targets[0]
            # logger.info("inputs len: %d" % len(input))
            # logger.info("%s" % (targets != FILL_VAL))
            mask = torch.where(targets != torch.tensor(FILL_VAL).cuda(),
                               torch.ones(targets.shape, dtype=torch.uint8).cuda(),
                               torch.zeros(targets.shape, dtype=torch.uint8).cuda())
            output = -1 * self.crf(cls_logits, targets, mask=mask, reduction='token_mean')
            # logger.info("output: %s" % str(output))
            # logger.info("output.shape: %s" % str(output.shape))
        else:
            output = cls_logits

        return output

    def predict(self, input):
        # logging.info("input in predict: %s" % str(input))
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        bilstm_out, _ = self.bilstm(hidden)
        cls_logits = self.clf_layer(tanh(bilstm_out))
        # if torch.isnan(cls_logits).any():
            # for i in range(input.shape[0]):
            #     logger.info("decoded input: %s" % str(TOKENIZER.decode(input[i, :])))

        # logger.info("hidden shape %s" % str(hidden.shape))
        # logger.info("bilstm_out.shape: %s" % str(bilstm_out.shape))
        # logger.info("tanh(bilstm_out): %s" % str(tanh(bilstm_out)))
        # logger.info("cls_logits.shape: %s" % str(cls_logits.shape))
        # logger.info("cls_logits: %s" % str(cls_logits))


        # logger.info("targets len: %d" % len(targets))
        # logger.info("targets[0] shape: %s" % str(targets[0].shape))
        # logger.info("targets[1] shape: %s" % str(targets[1].shape))
        # targets = targets[0]
        # logger.info("inputs len: %d" % len(input))
        # logger.info("%s" % (targets != FILL_VAL))
        # mask = torch.where(targets != torch.tensor(FILL_VAL).cuda(),
        #                    torch.ones(targets.shape, dtype=torch.uint8).cuda(),
        #                    torch.zeros(targets.shape, dtype=torch.uint8).cuda())
        # logger.info("mask: %s" % str(mask))
        # logger.info("mask.shape: %s" % str(mask.shape))
        # assert mask.shape[0] == 1
        # cls_logits_masked = []
        # new_masks = []
        # for b_idx in range(mask.shape[0]):
        #     num_valid = torch.sum(mask[b_idx, :])
        #     num_pad = mask.shape[1]-num_valid
        #
        #     clm = torch.masked_select(cls_logits[b_idx, :, :].t(), mask[b_idx, :]).reshape(num_valid, -1)
        #     logging.info("clm.shape: %s" % str(clm.shape))
        #     clm = torch.cat((clm, torch.zeros((num_pad, self.num_labels)).cuda()), 0).unsqueeze(0)
        #
        #     new_masks.append(torch.tensor([1]*int(num_valid) + [0]*int(num_pad)).unsqueeze(0).cuda())
        #     cls_logits_masked.append(clm)
        #
        # # cls_logits_masked = torch.masked_select(cls_logits, mask)
        # cls_logits_masked = torch.cat(cls_logits_masked, 0)
        # new_masks = torch.cat(new_masks, 0).byte()
        # output = self.crf.decode(cls_logits_masked, mask=new_masks)
        # output = self.crf.decode(cls_logits)

        output = self.crf.decode(cls_logits)
        return output

    def lm_forward(self, input):
        gpt_out = self.lm(input)
        hidden = gpt_out[0]
        # print(hidden.shape)
        lm_logits = self.lm_head(hidden)
        return lm_logits


# class CustomLoss(nn.Module):
#     def __init__(self, reduction: str = 'mean') -> None:
#         super(CustomLoss, self).__init__()
#         self.reduction = reduction
#
#
# class IdentityLoss(CustomLoss):
#     def __init__(self, reduction: str = 'mean') -> None:
#         super(IdentityLoss, self).__init__(reduction)
#
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         return self.reduction(input)
