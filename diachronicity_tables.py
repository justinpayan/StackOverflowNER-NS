import json
import random
import re
import sys

from collections import Counter, defaultdict

data_dir = "so_data"


def map_type(ent_type):
    if ent_type == "accuracy":
        return "Overall"
    else:
        return re.sub("_", "", ent_type)


if __name__ == "__main__":

    top_5_eps = {("_", "train"): [], ("_", "test"): [],
                 ("_temporal_", "train"): [], ("_temporal_", "test"): []}
    for skewtemp, trte in top_5_eps:
        for i in range(1, 6):
            with open(data_dir + ("/so%s%s_%d.json" % (skewtemp, trte, i))) as f:
                entity_type_cts = Counter()
                dset = json.load(f)
                for ex in dset:
                    for t in ex["tag_sequence"].strip().split():
                        if t.startswith("B"):
                            entity_type_cts[t] += 1
                top_5 = sorted(entity_type_cts.items(), key=lambda x: x[1], reverse=True)[:5]
                top_5 = [x[0] for x in top_5]
                top_5_eps[(skewtemp, trte)].append(top_5)

    skewtemp_to_formal = {"_temporal_": "Temporal", "_": "Skewed"}
    trte_to_formal = {"train": "Train", "test": "Test"}
    for skewtemp in ["_temporal_", "_"]:
        for trte in ["train", "test"]:
            for idx in range(5):
                to_print = ""
                if idx == 1:
                    to_print += skewtemp_to_formal[skewtemp]
                elif idx == 2:
                    to_print += trte_to_formal[trte]
                to_print += " & "
                to_print += " & ".join(["\\small " + map_type(x[idx][2:]) for x in top_5_eps[(skewtemp, trte)]])
                to_print += "\\\\"
                print(to_print)
            print("\\midrule")
    print(top_5_eps)