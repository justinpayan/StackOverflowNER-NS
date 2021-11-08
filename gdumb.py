import argparse
import json
import os
import random
import sys

from collections import Counter, defaultdict

data_dir = "so_data"


def extract_signature(example):
    sig = []
    for t in example['tag_sequence'].strip().split():
        if t.startswith("B"):
            sig.append(t[2:])
    return tuple(sig)


def add_to_dset(example, signature, dset, label_cts, sig_list, k):
    if len(dset) < k:
        # Add the new example
        dset.append(example)
        sig_list.append(signature)
        label_cts += Counter(set(signature))
    else:
        # Remove the example with the highest min label count (then highest 2nd lowest, etc), and add new example
        # Find the example with the highest min, highest 2nd min, etc
        sort_keys = []
        for idx, sig in enumerate(sig_list):
            cts = []
            for l in sig:
                cts.append(label_cts[l])
            sort_keys.append(tuple(sorted(cts)))
        sorted_idxs = [y[0] for y in sorted(sort_keys, key=lambda x: x[1:], reverse=True)]

        # Remove it
        dset.pop(sorted_idxs[0])
        remove_labels = set(sig_list.pop(sorted_idxs[0]))
        label_cts -= Counter(remove_labels)

        # Add the new example
        label_cts += Counter(set(signature))
        dset.append(example)
        sig_list.append(signature)

    return dset, label_cts, sig_list


def should_add(signature, label_cts, num_exs_per_label):
    labels = set(signature)

    for l in labels:
        if l not in label_cts or label_cts[l] < num_exs_per_label:
            return True
    return False


def gdumb_sample(rseed, k, split_type):
    random.seed(rseed)
    dset = []
    sig_list = []
    label_cts = Counter()

    if split_type == "temporal":
        prefix = "so_temporal_"
    elif split_type == "skewed":
        prefix = "so_"

    for ep in range(1, 6):
        with open(os.path.join(data_dir, prefix + "train_%d.json" % ep), 'r') as f:
            ep_dset = json.load(f)
            ep_dset = sorted(ep_dset, key=lambda x: random.random())
            for example in ep_dset:
                num_exs_per_label = (float(k) / len(label_cts)) if len(label_cts) else 10000000
                signature = extract_signature(example)
                if should_add(signature, label_cts, num_exs_per_label):
                    dset, label_cts, sig_list = add_to_dset(example, signature, dset, label_cts, sig_list, k)

    return dset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_type", default="temporal", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.split_type == "temporal":
        gdumb_prefix = "gdumb_t_"
        test_infix = "\"temporal_splits\",  \"so_temporal_\""
    elif args.split_type == "skewed":
        gdumb_prefix = "gdumb_"
        test_infix = "\"skewed_splits\",  \"so_\""
    else:
        print("split_type must be one of: [temporal, skewed]")
        sys.exit(0)

    for k in [300, 500, 1000, 1500, 1800]:
        for seed in range(10):
            dset = gdumb_sample(seed, k, args.split_type)
            for i in range(1, 6):
                with open(os.path.join(data_dir, gdumb_prefix + "%d_%d_%d.json" % (k, seed, i)), 'w') as f:
                    json.dump(dset, f)

    # Use this code to generate the dictionary items that go in settings.py
    for k in [300, 500, 1000, 1500, 1800]:
        for seed in range(10):
            for i in range(1, 6):
                print("\"" + gdumb_prefix + "%d_%d_%d\": {" % (k, seed, i))
                print("\t\"train\": os.path.join(args.data_dir, \"so_data\", \"gdumb\", \"" + gdumb_prefix + "%d_%d_%d.json\")," % (k, seed, i))
                print("\t\"eval\": os.path.join(args.data_dir, \"so_data\", " + test_infix + " + train_test + \"_%d.json\")," % i)
                print("\t\"test\": os.path.join(args.data_dir, \"so_data\", " + test_infix + " + train_test + \"_%d.json\")," % i)
                print("\t\"n_train_epochs\": 10")
                print("},")
