import json, os, sys

base_dir = sys.argv[1]
dset_name = sys.argv[2]

with open(os.path.join(base_dir, dset_name + ".json"), 'r') as f:
    new_data = []
    old_data = json.loads(f.read())
    for d in old_data:
        print(d)
        d["tag_sequence"] = " ".join([t if t != "O" else ("O-%s" % dset_name.split("_")[0])
                                      for t in d["tag_sequence"].split(" ")])
        new_data.append(d)

with open(os.path.join(base_dir, dset_name + "_O.json"), 'w') as f:
    json.dump(new_data, f)
