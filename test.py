import os
import json
from fp16 import FP16_Module
from collections import OrderedDict
from settings import TASK_DICT, MODEL_CONFIG, CONFIG_CLASS, CONFIG_NAME, LABEL_MAP, INVERSE_LABEL_MAP
from utils import NERDataset, create_dataloader, get_model_dir
from utils import lll_unbound_setting
from metrics import compute_metrics
from model import *
logger = logging.getLogger(__name__)


def test_one_to_one(task_load, task_eval, model, score_dict):

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))

    test_ner_data = NERDataset(TASK_DICT[task_eval]["test"], "test", SPECIAL_TOKEN_IDS[task_eval], task_eval).sort()
    test_dataloader = create_dataloader(test_ner_data, "test")
    n_examples = len(test_ner_data)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    ner_results = [0 for _ in range(n_examples)]
    if args.ic:
        intents = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    for n_steps, (_, ner_Xs, len_ner_Xs, Ys, _, _, _) in enumerate(test_dataloader):
        # assume n_gpus == 1
        # logger.info("len ner_Xs: %d" % len(ner_Xs))
        # if len(ner_Xs) > 1:
        #     for i in range(1, len(ner_Xs)):
        #         logger.info("extra data we are missing: %s" % str(ner_Xs[i].shape))
        for idx in range(len(ner_Xs)):
            ner_X = ner_Xs[idx]
            len_ner_X = len_ner_Xs[idx]
            Y = Ys[idx]
            n_inputs = ner_X.shape[0]

            # Construct a mask, and use the decode function
            if args.use_crf:
                # logger.info("ner_X.shape: %s" % str(ner_X.shape))
                # logger.info("Y.shape: %s" % str(Y.shape))
                # Y is just used for constructing the mask. Could use X, but it's easier to use Y.
                next_labels = model.predict(ner_X.cuda())
                # logging.info("next_labels: %s" % str(next_labels))
                if args.ic:
                    local_intents = [0 for _ in range(len(next_labels))]

                # logger.info("next_labels: %s" % str(next_labels))
                if args.use_task_in_ner:
                    for _idx in range(len(next_labels)):
                        next_labels[_idx] = next_labels[_idx][1:]

                if args.ic:
                    for _idx in range(len(next_labels)):
                        local_intents[_idx] = next_labels[_idx][0]
                        # logging.info("local_intents[_idx]: %s" % str(next_labels[_idx][0]))
                        next_labels[_idx] = next_labels[_idx][1:]

                for i in range(n_inputs):
                    max_tot_lens[cnt] = test_ner_data[cnt][1]
                    if args.ic:
                        intents[cnt] = local_intents[i]
                    ner_results[cnt] = next_labels[i]
                    cnt += 1
            else:
                pass

    for i in range(len(test_ner_data)):
        original, ner_X, _, Y, _, _, _, _ = test_ner_data[i]

        if args.ic:
            pred_intent = INVERSE_LABEL_MAP[intents[i]]
            # It's a bit confusing, we have to use Y[idx+1] to know if it's a fill_val, because Y has
            # the intent in the first position and the prediction doesn't.
            pred = [INVERSE_LABEL_MAP[l] for idx, l in
                    enumerate(ner_results[i]) if idx+1 < len(Y) and Y[idx+1] != FILL_VAL]

            intent_gold = INVERSE_LABEL_MAP[Y[0]]
            Y = [INVERSE_LABEL_MAP[l] for l in Y[1:] if l != FILL_VAL]

            intents[i] = [pred_intent, intent_gold]
            ner_results[i] = [original.split(" "), pred, Y]
        else:
            pred = [INVERSE_LABEL_MAP[l] for idx, l in enumerate(ner_results[i]) if idx < len(Y) and Y[idx] != FILL_VAL]

            Y = [INVERSE_LABEL_MAP[l] for l in Y if l != FILL_VAL]
            ner_results[i] = [original.split(" "), pred, Y]

    model_dir = model.model_dir
    conll_score_in_file = os.path.join(model_dir, "ner_conll_score_in_{}_{}".format(task_eval, "finish"))

    if task_eval in ['conll_eng', 'wnut', 'wnut_O'] or task_eval.startswith("so") or task_eval.startswith("gdumb"):
        with open(conll_score_in_file, 'w') as f:
            for original, pred, _y in ner_results:
                for o, p, y in zip(original, pred, _y):
                    f.write("%s %s %s\n" % (o, y, p))
                f.write("\n")

    elif args.ic:
        def is_other(tag):
            return tag.lower() == "o" or tag.lower() == "other"

        def add_bio(tag_seq):
            bio_tags = list()
            prev_tag = "O"
            for tag in tag_seq:
                if is_other(tag):
                    # Normalizing the Other tag to the one expected by conll_eval
                    tag = "O"
                else:
                    # If tag is already annotated with B/I we do nothing
                    if tag.startswith("B-") or tag.startswith("I-"):
                        tag = tag
                    # If the prev tag is None or Other or Different than current, we add B-. Otherwise I-.
                    elif is_other(prev_tag) or tag != prev_tag[2:]:
                        tag = "B-" + tag
                    else:
                        tag = "I-" + tag
                bio_tags.append(tag)
                prev_tag = tag
            return bio_tags

        # Convert each one to CoNLL format, write out to file, run conlleval.pl
        with open(conll_score_in_file, 'w') as f:
            for idx, (original, pred, _y) in enumerate(ner_results):
                f.write("%s %s %s\n" % ("__intent__", "B-%s" % intents[idx][1], "B-%s" % intents[idx][0]))
                for o, p, y in zip(original, add_bio(pred), add_bio(_y)):
                    f.write("%s %s %s\n" % (o, y, p))
                f.write("\n")

    conll_score_out_file = os.path.join(model_dir, "ner_conll_score_out_{}_{}".format(task_eval, "finish"))
    os.system('./conlleval.pl < %s > %s' % (conll_score_in_file, conll_score_out_file))

    return model, score_dict


def get_test_score(task_eval, ner_results, score_dict):
    score = compute_metrics(ner_results)
    score_dict[task_eval] = score


def test_one_to_many(task_load, already_tested):
    score_dicts = []
    if args.test_last_only:
        model_dir = get_model_dir([task_load])
        model_path = os.path.join(model_dir, 'model-{}'.format("finish"))
        config_path = os.path.join(model_dir, CONFIG_NAME)

        model_config = CONFIG_CLASS.from_json_file(config_path)
        lang_model = MODEL_CLASS(model_config).cuda().eval()
        state_dict = torch.load(model_path, map_location='cuda:0')
        model = NERModel(model_config.vocab_size, lm=lang_model, num_labels=len(LABEL_MAP)).cuda()
        model.load_state_dict(state_dict)

        if not args.fp32:
            model = FP16_Module(model)

        model.model_dir = model_dir
        logger.info("task: {}, epoch: {}".format(task_load, "finish"))
        score_dict = {k: None for k in args.tasks}
        with torch.no_grad():
            for task_eval in args.tasks:
                # if task_eval in already_tested or task_eval == task_load:
                test_one_to_one(task_load, task_eval, model, score_dict)
        logger.info("score: {}".format(score_dict))
        score_dicts.append(score_dict)

    else:
        eps = range(TASK_DICT[task_load]["n_train_epochs"])
        for ep in eps:
            model_dir = get_model_dir([task_load])
            model_path = os.path.join(model_dir, 'model-{}'.format(ep+1))
            config_path = os.path.join(model_dir, CONFIG_NAME)

            model_config = CONFIG_CLASS.from_json_file(config_path)
            lang_model = MODEL_CLASS(model_config).cuda().eval()
            state_dict = torch.load(model_path, map_location='cuda:0')
            model = NERModel(model_config.vocab_size, lm=lang_model, num_labels=len(LABEL_MAP)).cuda()
            model.load_state_dict(state_dict)

            if not args.fp32:
                model = FP16_Module(model)

            model.ep = ep
            model.model_dir = model_dir
            logger.info("task: {}, epoch: {}".format(task_load, ep+1))
            score_dict = {k: None for k in args.tasks}
            with torch.no_grad():
                for task_eval in args.tasks:
                    # if task_eval in already_tested or task_eval == task_load:
                    test_one_to_one(task_load, task_eval, model, score_dict)
            logger.info("score: {}".format(score_dict))
            score_dicts.append(score_dict)

    with open(os.path.join(model_dir, "metrics.json"),"w") as f:
        json.dump(score_dicts, f)


if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    init_logging(os.path.join(args.model_dir_root, 'log_test.txt'))
    logger.info('args = {}'.format(args))

    if args.seq_train_type == "multitask":
        test_one_to_many('_'.join(args.tasks))
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound, data_type="test", test_target="origin")
            for task_load in args.splitted_tasks:
                test_one_to_many(task_load)
        else:
            already_tested = set()
            for task_load in args.tasks:
                test_one_to_many(task_load, already_tested)
                already_tested.add(task_load)
