import json
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

        # sigs_to_cull = [sig for sig, v in dset.items() if len(v) == max_ct]
        # sig_to_cull = random.sample(sigs_to_cull, 1)[0]
        # ex_to_cull = random.sample(range(len(dset[sig_to_cull])), 1)[0]
        # dset[sig_to_cull].pop(ex_to_cull)
        # dset[signature].append(example)
    return dset, label_cts, sig_list


# def compute_label_cts_sig_list(dset):
#     sig_list = []
#     label_cts = Counter()
#     for ex in dset:
#         sig = extract_signature(ex)
#         sig_list.append(sig)
#         label_cts += Counter(set(sig))
#     return label_cts, sig_list


def should_add(signature, label_cts, num_exs_per_label):
    labels = set(signature)

    for l in labels:
        if l not in label_cts or label_cts[l] < num_exs_per_label:
            return True
    return False


def gdumb_sample(rseed, k):
    random.seed(rseed)
    dset = []
    sig_list = []
    label_cts = Counter()
    for ep in range(1, 6):
        with open(data_dir + ("/so_train_%d.json" % ep), 'r') as f:
            ep_dset = json.load(f)
            ep_dset = sorted(ep_dset, key=lambda x: random.random())
            for example in ep_dset:
                num_exs_per_label = (float(k) / len(label_cts)) if len(label_cts) else 10000000
                signature = extract_signature(example)
                if should_add(signature, label_cts, num_exs_per_label):
                    dset, label_cts, sig_list = add_to_dset(example, signature, dset, label_cts, sig_list, k)

    print("%d unique labels found" % len(label_cts))
    print("%d unique labels in dset" % len([v for _, v in label_cts.items() if v]))
    sorted_labels = sorted(label_cts.items(), key=lambda x: x[1])
    print(sorted_labels)
    print("%d is max number of examples for a label (%s)" % (sorted_labels[-1][1], sorted_labels[-1][0]))
    print("%d is min number of examples for a label (%s)" % (sorted_labels[0][1], sorted_labels[0][0]))
    return dset


if __name__ == "__main__":
    for k in [300, 500, 1000, 1500, 1800]:
        for seed in range(10):
            dset = gdumb_sample(seed, k)
            for i in range(5):
                with open(data_dir + "/gdumb_%d_%d_%d.json" % (k, seed, i), 'w') as f:
                    json.dump(dset, f)
