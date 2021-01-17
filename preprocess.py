import pickle as pkl
import re
import os
import json
import sys
from multiprocessing import Pool


# def serialize_data(redo=True):
#     print("serializing data ...")
#     for t in ["train", "eval", "test"]:
#         for task in TASK_DICT.keys():
#             data_path = TASK_DICT[task][t]
#             pkl_path = re.sub("json","pkl", data_path)
#             if os.path.exists(pkl_path) and not redo:
#                 continue
#             dataset = QADataset(data_path, t)
#             with open(pkl_path, "wb") as f:
#                 pkl.dump(dataset,f)
#     print("data serialized!")
#
#
# def dump_data_attrs(task):
#     attrs = {task:{"train":{}, "eval":{}, "test":{}}}
#     for t in ["train", "eval", "test"]:
#         print(task,t)
#         data_path = TASK_DICT[task][t]
#         pkl_path = re.sub("json","pkl", data_path)
#         with open(pkl_path, "rb") as f:
#             dataset = pkl.load(f)
#         attrs[task][t] = {"data_size": len(dataset),
#                           "max_a_len": dataset.max_a_len,
#         }
#     return attrs
#
#
# def parallel_dump_data_attrs(tasks=TASK_DICT.keys()):
#     print("creating data_attrs.json ...")
#     attr_dict = {}
#     with Pool(args.n_workers) as pool:
#         attrs = pool.map(dump_data_attrs, tasks)
#     for a in attrs:
#         attr_dict.update(a)
#     with open("data_attrs.json","w") as f:
#         json.dump(attr_dict,f)
#     print("data_attrs.json created!")


def get_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def combine_label_files(list_of_label_files, out_name):
    labels = {}
    idx = 0
    for fname in list_of_label_files.split(" "):
        with open(fname, 'r') as f:
            l = json.loads(f.read())
            for _l in l:
                if _l not in labels:
                    labels[_l] = idx
                    idx += 1
    with open(out_name, 'w') as f:
        json.dump(labels, f)


def snips_to_json(in_fname, out_train_fname, out_test_fname, label_fname):
    labels = {'O.AskUbuntu'}

    with open(in_fname, 'r') as f:
        js = json.loads(f.read())

    sentences = js["sentences"]

    idx = 0
    data = {'train': [],
            'test': []}
    for s in sentences:
        curr_sent = [t for t in re.split('([.\\-?() ])', s["text"]) if len(t.strip())]
        entity_map = {}
        for e in s["entities"]:
            for i in range(e["start"], e["stop"]+1):
                entity_map[i] = e["entity"]

        curr_tags = []
        for i in range(len(curr_sent)):
            if i in entity_map:
                curr_tags.append(entity_map[i])
                labels.add(entity_map[i])
            else:
                curr_tags.append("O.AskUbuntu")

        if s["training"]:
            to_append = data["train"]

        else:
            to_append = data["test"]

        to_append.append({'id': idx,
                          'sentence': " ".join(curr_sent),
                          'tag_sequence': " ".join(curr_tags)
                          })
        idx += 1

    print(len(data))

    with open(out_train_fname, 'w') as outf:
        json.dump(data["train"], outf)

    with open(out_test_fname, 'w') as outf:
        json.dump(data["test"], outf)

    label_dict = {}
    for idx, l in enumerate(labels):
        label_dict[l] = idx

    with open(label_fname, 'w') as labelf:
        json.dump(label_dict, labelf)


def wnut_to_json(in_fnames, out_fnames, label_fname):
    print(in_fnames)
    print(out_fnames)
    print(label_fname)

    def normalize_tag(tag):
        if len(tag) < 2:
            return tag
        elif tag[2:] == "geo-loc":
            return tag[:2] + "LOC"
        elif tag[2:] == "other":
            return tag[:2] + "MISC"
        elif tag[2:] == "person":
            return tag[:2] + "PER"
        else:
            return tag

    labels = set()
    for in_fname, out_fname in zip(in_fnames.split(" "), out_fnames.split(" ")):
        lines = get_lines(in_fname)
        idx = 0
        data = []
        curr_sent = []
        curr_tags = []

        for line in lines:
            if len(line.strip()) != 0:
                print(line.split("\t"))
                curr_sent.append(line.split("\t")[0].strip())
                tag = normalize_tag(line.split("\t")[1].strip())
                curr_tags.append(tag)
                labels.add(tag)
            elif len(curr_sent) > 0:
                data.append({'id': idx,
                             'sentence': " ".join(curr_sent),
                             'tag_sequence': " ".join(curr_tags)
                             })

                idx += 1
                curr_sent = []
                curr_tags = []

        print(len(data))

        with open(out_fname, 'w') as outf:
            json.dump(data, outf)

    label_dict = {}
    for idx, l in enumerate(labels):
        label_dict[l] = idx

    with open(label_fname, 'w') as labelf:
        json.dump(label_dict, labelf)


def conll_to_json(in_fnames, out_fnames, label_fname):
    labels = set()
    for in_fname, out_fname in zip(in_fnames.split(" "), out_fnames.split(" ")):
        lines = get_lines(in_fname)
        idx = 0
        data = []
        curr_sent = []
        curr_tags = []

        for line in lines[2:]:
            if len(line.strip()) != 0:
                curr_sent.append(line.split(" ")[0].strip())
                tag = line.split(" ")[3].strip()
                curr_tags.append(tag)
                labels.add(tag)
            elif len(curr_sent) > 0:
                data.append({'id': idx,
                             'sentence': " ".join(curr_sent),
                             'tag_sequence': " ".join(curr_tags)
                             })

                idx += 1
                curr_sent = []
                curr_tags = []

        print(len(data))

        with open(out_fname, 'w') as outf:
            json.dump(data, outf)

    label_dict = {}
    for idx, l in enumerate(labels):
        label_dict[l] = idx

    with open(label_fname, 'w') as labelf:
        json.dump(label_dict, labelf)


if __name__ == '__main__':
    # conll_to_json(sys.argv[1], sys.argv[2], sys.argv[3])
    # snips_to_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # wnut_to_json(sys.argv[1], sys.argv[2], sys.argv[3])
    # WNUT=/Users/juspayan/Downloads/twitter_nlp-master/data/annotated/wnut16/data
    # python preprocess.py "${WNUT}/train ${WNUT}/dev ${WNUT}/test" "train dev test" wnut_labels
    # combine_label_files(sys.argv[1], sys.argv[2])
    # python preprocess.py "wnut_labels conll_labels" wnut_conll_labels
# serialize_data()
# parallel_dump_data_attrs()
