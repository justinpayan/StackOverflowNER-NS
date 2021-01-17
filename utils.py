import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
from nltk.tokenize import word_tokenize
import pickle as pkl
import numpy as np
from copy import deepcopy
import os
from glob import glob
import logging
from fp16 import FP16_Module
import pathlib
from collections import OrderedDict
from settings import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from settings import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS, LABEL_MAP, INVERSE_LABEL_MAP
from multiprocessing import Pool
from twokenize import tokenizeRawTweetText
import sys
import random
import time
import quadprog
import io
import so_twokenize
import math
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_gen_token(task):
    if args.add_task_tokens:
        return '__' + task + '__'
    else:
        return '__gen__'


def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type != "multitask" else args.model_dir_root


def get_losses(model, ner_X, Y_train, gen_X, gen_Y, ner_loss_fct, lm_loss_fct):
    if "lll" in args.seq_train_type:
        # logger.info(ner_X[0][0])
        # logger.info(Y_train)

        # def get_len(one_d_tensor):
        #     l = 0
        #     for i in one_d_tensor:
        #         if i < 50258:
        #             l += 1
        #         else:
        #             return l

        # for i in range(ner_X[0][0].shape[0]):
            # l = get_len(ner_X[0][0][i,:])
            # logger.info(ner_X[0][0][i,:l])
            # logger.info("ner_X: %s" % str(TOKENIZER.decode(ner_X[0][0][i,:].tolist())))
            # logger.info("Y_train: %s" % str([INVERSE_LABEL_MAP[l] for l in Y_train[0][i,:].tolist()]))
        #     logger.info("gen_X: %s" % str(TOKENIZER.decode(gen_X[0][i,:l+1].tolist())))
        #     logger.info(gen_Y[0][i,:l+1].tolist())
        #     logger.info("gen_Y: %s" % str(TOKENIZER.decode(gen_Y[0][i,:l+1].tolist())))
        # logging.info("ner_X len: %s" % str(len(ner_X)))
        # logging.info("ner_X[0][0].shape: %s" % str(ner_X[0][0].shape))
        # logging.info("ner_X[0][0].type: %s" % str(ner_X[0][0].dtype))
        lm_logits = model.lm_forward(gen_X)
        lm_loss = lm_loss_fct([torch.transpose(l, 1, 2) for l in lm_logits], gen_Y)
        if args.use_crf:
            ner_loss = model(ner_X, targets=Y_train)
            # logger.info("ner_loss: %s" % str(ner_loss))
            # logger.info("lm_loss: %s" % str(lm_loss))
            return ner_loss[0], args.lm_lambda * torch.mean(lm_loss)
        else:
            ner_loss = ner_loss_fct([torch.transpose(l, 1, 2) for l in model(ner_X, targets=Y_train)], Y_train)
            return torch.mean(ner_loss), args.lm_lambda * torch.mean(lm_loss)
    else:
        if args.use_crf:
            ner_loss = model(ner_X, targets=Y_train)
            return ner_loss[0], torch.tensor(0.)
        else:
            ner_loss = ner_loss_fct([torch.transpose(l, 1, 2) for l in model(ner_X, targets=Y_train)], Y_train)
            return torch.mean(ner_loss), torch.tensor(0.)

def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len


def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)


def dynamic_collate_fn(data, batch_size):

    def local_collate():
        null_counter = 0
        _originals, _ner_Xs, _len_ner_Xs, _Ys, _Y_trains, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        max_len = max(len(data[j][1]) for j in range(st, ed))
        for j in range(st, ed):
            if None in data[j] or [] in data[j]:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = max_len - len(data[j][1])
            pad_len_gen = ner_max_len - len(data[j][5])

            # logger.info("%s\n%s\n%s\n%s\n%s\n"
            #             "%d\n%d\n%d\n%d" % (str(data[j][1]),
            #                                 str(data[j][2]),
            #                                 str(data[j][3]),
            #                                 str(data[j][4]),
            #                                 str(data[j][5]),
            #                                 max_len,
            #                                 pad_len,
            #                                 len(data[j][1]),
            #                                 len(data[j][3])))

            # logger.info("ner_max_len: %d" % ner_max_len)
            # logger.info("Y_max_len: %d" % Y_max_len)
            # logger.info("padded 1: %d" % len(pad_to_max_len(data[j][1], max_len - len(data[j][1]), SPECIAL_TOKEN_IDS["pad_token"])))
            # logger.info("padded 3: %d" % len(pad_to_max_len(data[j][3], Y_max_len - len(data[j][3]), FILL_VAL)))
            # logger.info("padded 4: %d" % len(pad_to_max_len(data[j][4], pad_len, SPECIAL_TOKEN_IDS["pad_token"])))
            # logger.info("padded 5: %d" % len(pad_to_max_len(data[j][5], pad_len, FILL_VAL)))
            _originals.append(data[j][0])
            _ner_Xs.append(pad_to_max_len(data[j][1], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _len_ner_Xs.append(data[j][2])
            _Ys.append(pad_to_max_len(data[j][3], pad_len, FILL_VAL))
            _Y_trains.append(pad_to_max_len(data[j][4], pad_len, FILL_VAL))
            _gen_Xs.append(pad_to_max_len(data[j][5], pad_len_gen, SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j][6], pad_len_gen, FILL_VAL))

        originals.append(_originals)
        ner_Xs.append(torch.tensor(_ner_Xs))
        len_ner_Xs.append(torch.tensor(_len_ner_Xs))
        Ys.append(torch.tensor(_Ys))
        Y_trains.append(torch.tensor(_Y_trains))
        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))

    originals, ner_Xs, len_ner_Xs, Ys, Y_trains, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    ner_max_len, cnt, st = 0, 0, 0

    for ed, datum in enumerate(data):
        ln = len(datum[5]) # use gen_X to calibrate
        # logger.info("len data: %d" % len(data))
        # logger.info("batch_size: %s" % batch_size)
        # logger.info("cnt: %d" % cnt)
        # logger.info(
        #     "max(ner_max_len, ln)**LEN_FACTOR * (ed - st + 1): %0.4f" % max(ner_max_len, ln) ** LEN_FACTOR * (
        #             ed - st + 1))
        # if max(ner_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
        if max(ner_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[0]:
            local_collate()
            cnt += 1
            ner_max_len = 0
            st = ed
        ner_max_len = max(ner_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return originals, ner_Xs, len_ner_Xs, Ys, Y_trains, gen_Xs, gen_Ys


def tokenize_sentence_with_tags(orig_sentence, sent_in, tags_in, intent=None):
    sentence = []
    if args.ic and intent:
        tag_sequence = [LABEL_MAP[intent.strip()]]
        tag_sequence_train = [LABEL_MAP[intent.strip()]]
    else:
        tag_sequence = []
        tag_sequence_train = []
    for w, t in zip(sent_in, tags_in):
        word_enc = TOKENIZER.encode(w)
        # logging.info(len(word_enc))
        # logging.info(len([label_map[t.strip()]] + [FILL_VAL] * (len(word_enc) - 1)))
        if len(word_enc) > 0:
            sentence.extend(word_enc)
            tag_sequence.extend([LABEL_MAP[t.strip()]] + [FILL_VAL] * (len(word_enc) - 1))
            tag_sequence_train.extend([LABEL_MAP[t.strip()]] * len(word_enc))
    return orig_sentence, sentence, tag_sequence, tag_sequence_train


class NERDataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, task_name, extra_data=[], extra_data_taskname=None):
        self.data_type = data_type
        self.label_map = LABEL_MAP
        self.gen_token = gen_token
        self.task_name = task_name
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)[:50] if args.short_exs_debug else json.load(f)

        self.data = []

        if len(raw_ds) > 0:
            self.data_tokenization(raw_ds)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x, extra_data_taskname), extra_data)
            extra_data = list(filter(lambda x: x, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            self.data += extra_data

    def etl_single_extra_data(self, data, task_name):
        # logging.info("data: %s" % str(data))
        if args.use_task_in_ner:
            gen_token_text = re.match("__(.*?)__", data[0].strip()).group()
        else:
            gen_token_text = get_gen_token(task_name)

        if args.ic:
            intent = data[1].strip().split(" ")[1].strip()

        rest_of_sentence = re.sub("__(.*?)__", "", data[0].strip()).strip()

        gen_token = TOKENIZER.convert_tokens_to_ids(gen_token_text)
        if args.ic:
            original, encoded_sentence, tag_sequence, tag_sequence_train = \
                tokenize_sentence_with_tags(rest_of_sentence, rest_of_sentence.split(" "),
                                            data[1].strip().split(" ")[2:], intent=intent)
        else:
            original, encoded_sentence, tag_sequence, tag_sequence_train = \
                tokenize_sentence_with_tags(rest_of_sentence, rest_of_sentence.split(" "),
                                            data[1].strip().split(" "))

        try:
            uid = uuid.uuid1().hex
            if args.ic:
                data = self.parse_example(gen_token, gen_token_text, original,
                                         encoded_sentence, tag_sequence, tag_sequence_train, uid, intent=intent)
            else:
                data = self.parse_example(gen_token, gen_token_text, original,
                                         encoded_sentence, tag_sequence, tag_sequence_train, uid)
            # logging.info("processed data: %s" % str(data))
        except ValueError:
            return
        return data


    @staticmethod
    def concat_example(gen_token, intent_token, sentence, eos_token):
        if len(sentence) + 1 > args.max_len:
            logger.warning('a sentence with len {} is too long!'.format(len(sentence) + 1))
            return
        example = gen_token + intent_token + sentence[:args.max_len - 1] + eos_token
        return example

    @staticmethod
    def sample_entity_tokens(tag_seq):
        sample_set = set()
        for t in tag_seq:
            if t != FILL_VAL:
                label = INVERSE_LABEL_MAP[t]
                if label[0] != "O":
                    sample_set.add(label[2:])
        entity_tokens = []
        for l in sample_set:
            entity_tokens.extend(TOKENIZER.encode("__%s__" % l))
        return entity_tokens


    def parse_example(self, gen_token, task_specific_token_label, original, encoded_sentence, tag_sequence, tag_sequence_train, idx, intent=None):
        if args.ic:
            intent_prompt = TOKENIZER.encode("__%s__" % intent)
            intent_token = [SPECIAL_TOKEN_IDS["ic"]]
        else:
            intent_prompt = []
            intent_token = []

        if args.use_task_in_ner:
            ner_example = self.concat_example([gen_token], intent_token, encoded_sentence, [])
            # intent will be included in tag_sequence
            Y_example_train = self.concat_example([LABEL_MAP[task_specific_token_label]], [], tag_sequence_train, [])
        elif args.add_task_tokens:
            ner_example = self.concat_example([gen_token], intent_token, encoded_sentence, [])
            Y_example_train = self.concat_example([LABEL_MAP["O"]], [], tag_sequence_train, [])
        else:
            ner_example = self.concat_example([], intent_token, encoded_sentence, [])
            Y_example_train = tag_sequence_train

        # logging.info("ner_example: %s" % str(ner_example))
        # logging.info("Y_example_train: %s" % str(Y_example_train))

        Y_example = tag_sequence
        # Y_example = [self.label_map[l.strip()] for l in tag_sequence.split(" ")]

        # if args.add_ent_tokens:
        #     entity_tokens = self.sample_entity_tokens(tag_sequence)
        #     gen_X_start = [gen_token] + entity_tokens + TOKENIZER.encode("__start__")
        #     gen_Y_start = entity_tokens + TOKENIZER.encode("__start__")
        # else:
        #     gen_X_start = [gen_token]
        #     gen_Y_start = []
        gen_X_start = [gen_token]
        gen_Y_start = []

        # gen_X_example = self.concat_example(gen_X_start, intent_token, encoded_sentence, [])
        # Just change it so that when you are doing IC, you have the intent on the INPUT. So then we can
        # control the intent but we don't have to change the NER input, or otherwise change the gen output.
        gen_X_example = self.concat_example(gen_X_start, intent_prompt, encoded_sentence, [])
        gen_Y_example = self.concat_example(gen_Y_start, intent_token, encoded_sentence, [self.eos_token])

        return original, ner_example, len(ner_example), Y_example, Y_example_train, gen_X_example, gen_Y_example, idx


    def parallel_tokenization(self, d):
        examples = []
        # sentence = TOKENIZER.encode(d["sentence"])
        # sentence = TOKENIZER.convert_tokens_to_ids(d["sentence"].split(" "))
        # tag_sequence = d["tag_sequence"]

        # Following utils_ner.py from HuggingFace, line ~300
        if args.ic:
            original, encoded_sentence, tag_sequence, tag_sequence_train = \
                tokenize_sentence_with_tags(d["sentence"], d["sentence"].split(" "), d["tag_sequence"].split(" "),
                                            intent=d["intent"])
        else:
            original, encoded_sentence, tag_sequence, tag_sequence_train = \
                tokenize_sentence_with_tags(d["sentence"], d["sentence"].split(" "), d["tag_sequence"].split(" "))

        # original, encoded_sentence, tag_sequence, tag_sequence_train = self.tokenize_sentence_with_tags(d)
        # logging.info("sentence and tag sequence before:\n%s\n%s\n"
        #              "sentence and tag sequence after:\n%s\n%s" % (str(d["sentence"].split(" ")),
        #                                                            str(d["tag_sequence"].split(" ")),
        #                                                            str(encoded_sentence),
        #                                                            str(tag_sequence)))
        id = d["id"]
        if args.ic:
            examples.append(self.parse_example(self.gen_token, "__%s__" % self.task_name, original,
                                            encoded_sentence, tag_sequence, tag_sequence_train, id, intent=d["intent"]))
        else:
            examples.append(self.parse_example(self.gen_token, "__%s__" % self.task_name, original,
                                               encoded_sentence, tag_sequence, tag_sequence_train, id))
        return examples


    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        for datum in data:
            self.data.extend(datum)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    # def _sort_by_index(self,data):
    #    datum = []
    #    for d in data:
    #        for qa in d["qas"]:
    #            datum.append({"context":d["context"], "qas":[qa]})
    #    datum.sort(key=lambda x:x["qas"][0]["id"])
    #    return datum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class EarlyStopping:
    def __init__(self, logger,  patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(model_dir)
        TOKENIZER.save_pretrained(model_dir)
        self.val_loss_min = val_loss


class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, loss, scheduler_steps):
        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

        if "gem" in args.seq_train_type and self.model.task_id >0: 
            store_grad(self.model.parameters, self.model.grads, self.model.grad_dims,self.model.task_id)
            indx = torch.cuda.LongTensor([i for i in range(self.model.task_id)])
            dotp = torch.mm(self.model.grads[:, self.model.task_id].unsqueeze(0),
                            self.model.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.model.grads[:, self.model.task_id].unsqueeze(1),
                              self.model.grads.index_select(1, indx), args.qp_margin)
                # copy gradients back
                overwrite_grad(self.model.parameters,
                               self.model.grads[:, self.model.task_id],
                               self.model.grad_dims)
            
        if args.seq_train_type in args.REG_TYPE_KEYS:
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()


class GEMStep:
    def __init__(self, model, parallel_model, train_loss_fct, optimizer):
        self.model = model
        self.parallel_model = parallel_model
        self.train_loss_fct = train_loss_fct
        self.optimizer = optimizer

    def __call__(self,current_task_id):
        for past_task_id, md in enumerate(args.memory_data):
            # Not saving current task's grads.
            if past_task_id >= current_task_id: return
            qadata = QADataset(None, "test", "gen", md)
            dataloader = create_dataloader(qadata, "test")
            grads_tmp = torch.zeros(sum(self.model.grad_dims),).cuda()
            if not args.fp32:
                grads_tmp = grads_tmp.half() 
            for _, _, cqa, _, Y, gen_X, gen_Y in dataloader:
                #CHECK
                n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
                self.optimizer.zero_grad()
                for i in range(len(cqa)):
                    cqa[i] = (cqa[i].to(args.device_ids[i]),)
                    Y[i] = Y[i].to(args.device_ids[i])
                    gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                    gen_Y[i] = gen_Y[i].to(args.device_ids[i])

                losses = get_losses(self.model, cqa, Y, gen_X, gen_Y, self.train_loss_fct)
                loss = sum(losses)
                if not args.fp32:
                    self.optimizer.backward(loss, update_master_grads=False)
                else:
                    loss.backward()

                if not args.fp32:
                    #copy fp16 grads to fp32 grads  
                    self.optimizer.update_master_grads()
                    self.optimizer.clip_master_grads(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                i = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        beg = 0 if i == 0 else sum(self.model.grad_dims[:i])
                        end = sum(self.model.grad_dims[:i+1])
                        grads_tmp[beg: end] += param.grad.data.view(-1)*n_inputs
                    i += 1

            grads_tmp /= len(qadata)
            self.model.grads[:, past_task_id].copy_(grads_tmp)
            self.optimizer.zero_grad()


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if args.debug or self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][3])
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=args.n_workers,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs[0]


def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(MODEL_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0


def sample_sequence(model, need_process, ner_results, all_pasts, max_tot_lens):
    lm = model.lm
    while len(need_process) > 0:
        # logging.info("len(need_process): %d" % len(need_process))
        # logging.info("need_process: %s" % str(need_process))
        first_id = next(iter(need_process))
        shortest_len = len(ner_results[first_id])
        decode_batch_size = int(args.memory_sizes[0] * MEMORY_FACTOR[args.seq_train_type] // (shortest_len+1)**LEN_FACTOR)
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            # batch_ids, input_ids, past = [], [], [[] for _ in range(MODEL_CONFIG.n_layer)]
            batch_ids, _input_ids, pasts = {}, {}, {}
            while True:
                try:
                    cur_id = next(it)
                    # logging.info("len(ner_results[cur_id]): %d, shortest_len: %d" % (len(ner_results[cur_id]), shortest_len))
                    if len(ner_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    seq_len = all_pasts[0][cur_id].shape[2]
                    if seq_len not in _input_ids:
                        batch_ids[seq_len] = []
                        _input_ids[seq_len] = []
                        pasts[seq_len] = [[] for _ in range(MODEL_CONFIG.n_layer)]
                    batch_ids[seq_len].append(cur_id)
                    if args.model_name == "gpt2":
                        if args.ic and seq_len == 0:
                            _input_ids[seq_len].append(ner_results[cur_id][-2:])
                        else:
                            _input_ids[seq_len].append(ner_results[cur_id][-1:])
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            pasts[seq_len][layer_id].append(all_pasts[layer_id][cur_id])
                    else:
                        _input_ids[seq_len].append(ner_results[cur_id])
                    if max([len(iids) for iids in _input_ids.values()]) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = sum([len(iids) for iids in _input_ids.values()])
            if n_inputs == 0:
                break

            for seq_len in batch_ids:
                input_ids = torch.stack(_input_ids[seq_len])
                # logging.info("seq_len: %d" % seq_len)
                past = pasts[seq_len]
                # logging.info("past: %s" % str(past))
                # for idx, i in enumerate(past[0]):
                #     logging.info("past[0][%d].shape: %s" % (idx, str(i.shape)))
                if args.model_name == "gpt2":
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        # for tens in past[layer_id]:
                        #     seq_len = tens.shape[2]
                        #     if seq_len not in pasts_indexed_by_seq_len:
                        #         pasts_indexed_by_seq_len[seq_len] = [[] for _ in range(MODEL_CONFIG.n_layer)]
                        #     pasts_indexed_by_seq_len[seq_len][layer_id].append()
                        #
                        # logging.info("sequence lens in past: %s" % (str(pasts_indexed_by_seq_len.keys())))

                        past[layer_id] = torch.stack(past[layer_id], dim=1)
                    # logging.info("input_ids.shape: %s, past[0].shape: %s" % (str(input_ids.shape), str(past[0].shape)))
                    all_gpt_outputs, new_pasts = lm(input_ids=input_ids.cuda(), past=past)
                else:
                    all_gpt_outputs, new_pasts = lm(input_ids=input_ids.cuda())

                # logging.info("input_ids.shape: %s" % str(input_ids.shape))
                # logging.info("all_gpt_outputs.shape: %s" % str(all_gpt_outputs.shape))
                # logging.info("len(pasts): %d" % len(pasts))
                # for p in range(len(pasts)):
                #     logging.info("pasts[%d].shape: %s" % (p, str(pasts[p].shape)))

                outputs = model.lm_head(all_gpt_outputs)

                # logging.info("outputs.shape: %s" % str(outputs))
                # outputs = all_outputs[0]
                # if args.model_name == "gpt2":
                #     pasts = all_outputs[1]

                next_logits = outputs[..., -1, :] / args.temperature_ner
                next_tokens = logits_to_tokens(next_logits).cpu()

                # logging.info("ner_results: %s" % str(ner_results))
                # logging.info("next_tokens: %s" % str(next_tokens))

                for i, cur_id in enumerate(batch_ids[seq_len]):
                    if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                        remove_ids.append(cur_id)
                    else:
                        # logging.info("cur_id: %s, i: %s" % (str(cur_id), str(i)))
                        ner_results[cur_id] = torch.cat((ner_results[cur_id], next_tokens[i]))
                        limit = 10 if args.short_exs_debug else args.max_len
                        if len(ner_results[cur_id]) in [max_tot_lens[cur_id], limit]:
                            remove_ids.append(cur_id)
                        elif args.model_name == "gpt2":
                            for layer_id in range(MODEL_CONFIG.n_layer):
                                all_pasts[layer_id][cur_id] = new_pasts[layer_id][:, i].type(torch.float if args.fp32 else torch.half)
        logging.info("Removing ids: %s" % str(remove_ids))
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)


def write_extra_data(dump_path, ner_results):
    logger.info(f"writing extra data in {dump_path} ...")
    with open(dump_path,"w",newline="",encoding="utf-8") as f:
        lm_writer = csv.writer(f,delimiter=',')
        lm_writer.writerow(["gen"])
        for l in ner_results:
            lm_writer.writerow(l)


def parse_single_real_data(data, task):
    sentence = data["sentence"]
    tag_seq = data["tag_sequence"]
    if args.ic:
        sentence = SPECIAL_TOKENS["ic"] + " " + sentence
        tag_seq = data["intent"] + " " + tag_seq

    if args.use_task_in_ner:
        # return [SPECIAL_TOKENS[task] + " " + sentence, SPECIAL_TOKENS[task] + " " + tag_seq]
        return [SPECIAL_TOKENS[task] + " " + sentence, tag_seq]
    else:
        return [sentence, tag_seq]


def get_real_data(task, train_extra_data, accum=True, encode=True):
    task_idx = args.tasks.index(task)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    if accum:
        prev_tasks = args.tasks[:task_idx]
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))//len(prev_tasks)
    else:
        prev_tasks = [args.tasks[task_idx-1]]
        gen_size = int(gen_size * args.gen_lm_sample_percentage)

    datum = []
    for prev_task in prev_tasks:
        with open(TASK_DICT[prev_task]["train"],"r") as f:
            data = json.load(f)
        indices = np.random.choice(range(len(data)), gen_size)
        for i in indices:
            d = parse_single_real_data(data[i],prev_task)
            datum.append(d)
            if encode:
                train_extra_data.append(TOKENIZER.encode(d))
            else:
                train_extra_data.append(d)
        
    model_dir = get_model_dir([prev_task])
    dump_path = os.path.join(model_dir,"real.csv")
    write_extra_data(dump_path, datum)
    return dump_path


def read_extra_data(gen_path, train_extra_data):
    with open(gen_path,"r") as lm_file:
        reader = csv.reader(lm_file,delimiter=',')
        next(reader)
        for row in reader: 
            row = [row[0].strip(), row[1].strip()]
            train_extra_data.append(row)


def determine_etd(task_name):
    # Load train data and count up combinations of entity types
    entity_type_combos = []
    with open(TASK_DICT[task_name]["train"], 'r') as data_path:
        raw_ds = json.load(data_path)
        for d in raw_ds:
            tag_seq = d["tag_sequence"].split(" ")
            entity_type_combo = set([l[2:] for l in tag_seq if l[0] != "O"])
            if len(entity_type_combo) > 0:
                entity_type_combo = tuple(sorted(entity_type_combo))
                entity_type_combos.append(entity_type_combo)
    return entity_type_combos


def sample_entity_type_tokens(distrib):
    entity_type_combo = random.sample(distrib, 1)[0]
    # logging.info("distrib: %s" % str(distrib))
    # logging.info("entity_type_combo: %s" % str(entity_type_combo))
    entity_type_tokens = [("__%s__" % ent_type) for ent_type in entity_type_combo]
    # logging.info("entity_type_tokens: %s" % entity_type_tokens)
    entity_type_token_ids = TOKENIZER.convert_tokens_to_ids(entity_type_tokens)
    # logging.info("entity_type_token_ids: %s" % entity_type_token_ids)
    return entity_type_token_ids


def create_extra_data(task, prev_task, model, train_extra_data):
    if args.real_sample:
        logger.info(f"using real data as extra data")
        return get_real_data(task, train_extra_data, encode=False)
    task_cnt = args.tasks.index(task)
    model_dir = get_model_dir([prev_task])
    gen_path = os.path.join(model_dir,"lm.csv")
    # if os.path.exists(gen_path):
    #     logger.info(f"extra data exists in {gen_path}, read it!")
    #     return read_extra_data(gen_path, train_extra_data)
    # gen_size = DATA_ATTRS[task]["train"]["data_size"]
    # gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    # gen_size -= (gen_size % task_cnt)

    # Get the intent distribution
    if args.ic:
        intent_cts = Counter()
        with open(TASK_DICT[prev_task]["train"], "r") as f:
            data = json.load(f)
        for d in data:
            intent_cts[d['intent'].strip()] += 1
        intents = sorted(intent_cts.keys())
        intent_distrib = [float(intent_cts[i])/len(data) for i in intents]

    # Generate as much as the original data size, then sample
    gen_size = DATA_ATTRS[prev_task]["train"]["data_size"]

    if args.short_exs_debug:
        gen_size = 100 - (100 % task_cnt)

    logging.info("gen_size: %s" % gen_size)

    # if args.debug:
    #     gen_size = task_cnt

    model.eval()

    need_process = OrderedDict()
    ner_results = []

    # Just generate for the most recent task
    task_name = prev_task
    if args.ic:
        num_each_intent = [math.floor(i_prop*gen_size) for i_prop in intent_distrib]
        for intent, amt in zip(intents, num_each_intent):
            ner_results.extend([torch.tensor([SPECIAL_TOKEN_IDS[task_name], TOKENIZER.encode("__%s__" % intent)[0]]) for _ in range(amt)])
        gen_size = len(ner_results)
    else:
        ner_results.extend([torch.tensor([SPECIAL_TOKEN_IDS[task_name]]) for _ in range(gen_size)])
    all_pasts = [[
        torch.empty(2, MODEL_CONFIG.n_head, 0, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head,
                    dtype=torch.float if args.fp32 else torch.half).cuda()
        for _ in range(gen_size)
    ] for __ in range(MODEL_CONFIG.n_layer)]

    max_tot_lens = [args.max_len for _ in range(gen_size)]

    for i in range(gen_size):
        need_process.update([[i, None]])
        if len(need_process) > int(args.memory_sizes[0] * 0.12):
            sample_sequence(model, need_process, ner_results, all_pasts, max_tot_lens)
    sample_sequence(model, need_process, ner_results, all_pasts, max_tot_lens)

    _ner_results = [res.tolist() for res in ner_results]
    if args.add_ent_tokens:
        logging.info("ner_results, before filter: %s" % str([" ".join(re.split('([.\\-?() ])', TOKENIZER.decode(res))) for res in _ner_results]))
        for idx in range(len(_ner_results)):
            _ner_results[idx] = [_ner_results[idx][0]] + _ner_results[idx][num_ent_type_tokens[idx] + 2:]
    # train_extra_data.extend(ner_results)
    # ner_results = [TOKENIZER.decode(res) for res in _ner_results]
    # for any of them that are actually supposed to be for ubuntu, can we shift the period over so that
    # is actually its own word? But it seems that this would ruin things for conll?
    # Just do it for now; and then we can go back and change to the regular way when we move to WNUT.

    # replace with the correct tokenizer, depending on the dataset.
    ner_results = []
    for res in _ner_results:
        if 'conll_eng' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['conll_eng']:
            unprocessed = TOKENIZER.decode(res[1:]).strip()
            new_res = word_tokenize(unprocessed)
            first_num = re.match("[0-9]+\.", unprocessed)
            if first_num:
                ner_results.append(SPECIAL_TOKENS['conll_eng'] + " " + first_num.group() + " " + " ".join(new_res[2:]))
            else:
                ner_results.append(SPECIAL_TOKENS['conll_eng'] + " " + " ".join(new_res))
        elif ('wnut' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['wnut']) \
            or ('wnut_O' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['wnut_O']):
            unprocessed = TOKENIZER.decode(res).strip()
            ner_results.append(" ".join(tokenizeRawTweetText(unprocessed)))
        elif ('so_1' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['so_1']) or \
                ('so_2' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['so_2']) or \
                ('so_3' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['so_3']) or \
                ('so_4' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['so_4']) or \
                ('so_5' in SPECIAL_TOKEN_IDS and res[0] == SPECIAL_TOKEN_IDS['so_5']):
            unprocessed = TOKENIZER.decode(res).strip()
            try:
                ner_results.append(" ".join(so_twokenize.tokenize(unprocessed)))
            except:
                pass
        else:
            if args.ic:
                unprocessed = TOKENIZER.decode(res).strip().split(" ")
                unprocessed[1] = "__intent__"
                ner_results.append(" ".join(unprocessed))
            else:
                unprocessed = TOKENIZER.decode(res).strip()
                ner_results.append(unprocessed)

    # ner_results = [" ".join(re.split('([.\\-?() ])', TOKENIZER.decode(res))) for res in _ner_results]

    logging.info("ner_results: %s" % str(ner_results))

    for idx in range(len(ner_results)):
        gen_tokens =  set([SPECIAL_TOKEN_IDS[task_name] for task_name in args.tasks[:task_cnt]])
        sentence = []
        mask = []
        logging.info("generated sentence: %s" % str(ner_results[idx]))
        for w in ner_results[idx].strip().split(" "):
            # logging.info("word: %s" % w)
            word_enc = TOKENIZER.encode(w.strip())
            # logging.info("len word_enc: %d" % len(word_enc))
            if len(word_enc) > 0:
                if word_enc[0] not in gen_tokens or args.use_task_in_ner:
                    sentence.extend(word_enc)
                    mask.extend([1] + [0] * (len(word_enc) - 1))
        input_tensor = torch.tensor(sentence).unsqueeze(0).long().cuda()
        # logging.info("input_tensor.shape: %s" % str(input_tensor.shape))
        # logging.info("input_tensor.type: %s" % str(input_tensor.dtype))
        labels_out = model.predict(input_tensor)
        logging.info("labels out: %s" % str(labels_out))
        if args.use_task_in_ner and not args.ic:
            for _idx in range(len(labels_out)):
                labels_out[_idx] = labels_out[_idx][1:]
                mask = mask[1:]
        logging.info("labels out: %s" % str(labels_out))
        # labels_out = logits_to_labels(tag_ids_out)
        pred = [INVERSE_LABEL_MAP[l] for i, l in enumerate(labels_out[0]) if i < len(mask) and mask[i]]
        # logging.info("pred: %s" % str(pred))

        # _ner_results[idx] = (_ner_results[idx], pred)
        ner_results[idx] = [ner_results[idx], " ".join(pred)]

    write_extra_data(gen_path, ner_results)

    # Now instead of just adding the generated data to the training data, pull from the respective files.
    # And we will HERE sample the right number of them.
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    gen_size -= (gen_size % task_cnt)

    for task in args.tasks[:task_cnt]:
        model_dir = get_model_dir([task])
        gen_path = os.path.join(model_dir, "lm.csv")

        extra_data = []
        read_extra_data(gen_path, extra_data)
        amount_to_sample = min(gen_size//task_cnt, len(extra_data))
        extra_data = random.sample(extra_data, amount_to_sample)
        train_extra_data.extend(extra_data)

    # train_extra_data.extend(ner_results)
    logging.info("train_extra_data: %s" % str(train_extra_data))
    # logging.info("final ner_results: %s" % str(ner_results))

    model.train()


def logits_to_tokens(next_logits):
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_ner, top_p=args.top_p_ner)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens


def logits_to_labels(next_logits):
    # filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_ner, top_p=args.top_p_qa)
    # log_probs = F.softmax(next_logits, dim=-1)
    labels = torch.argmax(next_logits, dim=-1)
    return labels

 
def lll_unbound_setting(split_size=10,data_type="train",test_target="self"):
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    if data_type == "test":
        args.splitted_tasks = [f"task_{i}" for i in range(split_size)]
        args.n_train_epochs = {task: args.n_train_epochs for task in args.splitted_tasks}
        if test_target in ["self","all"]:
            for no in range(split_size):  
                task = f"task_{no}" 
                test_data_path = os.path.join(data_dir,f"{task}-test.json")
                TASK_DICT[task] = {}
                TASK_DICT[task]["test"] = test_data_path
            if test_target == "all":
                args.tasks += args.splitted_tasks
            else:
                args.tasks = args.splitted_tasks
    elif data_type == "train":
        create_lll_unbound_data(split_size)
        args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return TASK_DICT


def create_lll_unbound_data(split_size=10): 
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    datum = [] 
    test_datum = []
    data_sizes = [] 
    chunk_sizes = []
    for task in args.tasks:
        train_data_path = TASK_DICT[task]["train"]
        with open(train_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            data_sizes.append(len(data))
            datum += data
        test_data_path = TASK_DICT[task]["test"]
        with open(test_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            test_datum.append(data) 
    chunk_size = int(np.ceil(len(datum)/split_size))

    tasks = []
    for no, i in enumerate(range(0, len(datum), chunk_size)):  
        task = f"task_{no}" 
        tasks.append(task)
        chunk = datum[i:i + chunk_size] if i < len(datum)-chunk_size else datum[i:]
        chunk_sizes.append(len(chunk))
        DATA_ATTRS[task] = {"train":{"data_size":None}}
        DATA_ATTRS[task]["train"]["data_size"] = len(chunk)
        train_data_path = os.path.join(data_dir,f"{task}-train.json")
        with open(train_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task] = {}
        TASK_DICT[task]["train"] = train_data_path
    args.tasks = tasks

    sis = get_split_indices(data_sizes,chunk_sizes)
    test_split = []
    for dic in sis.values():
        merged_data = []
        for k,v in dic.items():
            from_index = int(len(test_datum[k])*v[0])
            to_index = int(len(test_datum[k])*v[1])
            merged_data+= test_datum[k][from_index:to_index]
        test_split.append(merged_data)

    for no, chunk in enumerate(test_split):  
        task = f"task_{no}" 
        test_data_path = os.path.join(data_dir,f"{task}-test.json")
        with open(test_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task]["test"] = test_data_path


def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para: 
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs":[d]})
    return datum


def get_split_indices(data_sizes,chunk_sizes):
    ds = deepcopy(data_sizes)
    records = {}
    tmp = {}
    order = 0 # data_sizes index
    i = 0 # chunk_sizes index
    while len(data_sizes)>0:
        d0 = data_sizes[0]
        c0 = chunk_sizes[0]
        if d0>c0:
            val = c0/ds[order]
        else:
            val = d0/ds[order]

        if order not in tmp:
            rec = (0,val)
            tmp[order] = val
        else:
            rec = (tmp[order],tmp[order]+val)
            tmp[order] += val
        if i in records:
            records[i][order] = rec
        else:
            records[i] = {order: rec}

        if d0>c0:
            data_sizes[0]-=c0
            chunk_sizes.pop(0)
            i+=1
        else:
            chunk_sizes[0]-=d0
            data_sizes.pop(0)
            order+=1
            if d0==c0:
                chunk_sizes.pop(0)
                i+=1
    return records


def store_grad(get_ps, grads, grad_dims, task_id): 
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[:i+1])
            grads[beg: end, task_id].copy_(param.grad.data.view(-1))
        i += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
