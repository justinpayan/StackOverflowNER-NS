import argparse, csv, datetime, json, os, random, re, sys
import xml.etree.ElementTree as ET
from so_twokenize import tokenize
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

punkt_param = PunktParameters()
abbreviation = ['u.s.a', 'fig', 'etc', 'eg', 'mr', 'mrs', 'e.g', 'no', 'vs', 'i.e']
punkt_param.abbrev_types = set(abbreviation)
sent_tokenizer = PunktSentenceTokenizer(punkt_param)

np_gen = np.random.default_rng(seed=31415)


def get_lines(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            yield line


def is_same(qa, text):
    # if len(qa) != len(text):
    #     return False
    try:
        for idx, l in enumerate(text):
            if l[0].startswith("CODE_BLOCK"):
                if qa[idx][0] != "CODE_BLOCK":
                    return False
                else:
                    # print("matching code blocks, continue")
                    return True
            # The tokenizer is so different from what was actually used, the best we can do is see if a majority of lines
            # match. Because the qid matches anyway, this is probably good enough.
            if qa[idx] is not None and len(l) == len(qa[idx]):
                for local_idx, s in enumerate(l):
                    # print(s.split("\t")[0])
                    # print(qa[idx][local_idx])
                    if qa[idx][local_idx] != s.split("\t")[0]:
                        return False
                return True
    except:
        return False


def get_create_date(qid, text, from_xml_tokens):
    # print(qid)
    # print(qid in from_xml_tokens.keys())
    # print(from_xml_tokens[qid])
    qas = from_xml_tokens[qid]
    # print(text)
    for qa in qas:
        # print(qa[1])
        if is_same(qa[1], text):
            return qa[0]
    print("no match")
    print("\n".join([" ".join([t.split("\t")[0] for t in sent]) for sent in text]))
    print("\n")
    print(qas)
    print("\n\n")
    if len(qas) > 0:
        return qas[0][0]


def create_json_elts(text, create_date, qid, start_id):
    json_elts = []
    curr_id = start_id
    new_labels = []

    for l in text:
        if not l[0].startswith("CODE_BLOCK"):
            json_elts.append({
                "id": curr_id,
                "sentence": " ".join([s.split("\t")[0] for s in l]),
                "tag_sequence": " ".join([s.split("\t")[1] for s in l]),
                "question_id": qid
            }
            )
            if create_date is not None:
                json_elts[-1]["creation_date"] = create_date.strftime("%Y-%m-%d")
            curr_id += 1

            new_labels_here = set()
            for label in [s.split("\t")[1] for s in l]:
                new_labels_here.add(label)
            new_labels.append(new_labels_here)

    return json_elts, new_labels, curr_id


def load_xml_dump(xml_dump_dir):
    from_xml_tokens = defaultdict(list)
    skipped_sentences = 0
    nonskipped_sentences = 0
    for idx, file in enumerate(os.listdir(xml_dump_dir)):
        # if idx % 1 == 0:
        print(float(idx) / len(os.listdir(xml_dump_dir)))
        lines = list(get_lines(os.path.join(xml_dump_dir, file)))
        qid = lines[0].strip().split(" ")[-1].split("_")[0]
        creation_date = datetime.datetime.strptime(lines[2][:10], "%Y-%m-%d")
        text = []
        # print("STARTING TOKENIZING:")
        # print(lines[4:])
        # print("END.")
        # print("year: %d" % creation_year)
        # print()
        # print(lines)
        for l in lines[4:]:
            try:
                text.append(tokenize(l))
            except Exception as e:
                print(e)
                print(qid)
                print(l)
                skipped_sentences += 1
                text.append(None)
        if len([t for t in text if t is not None]) > 0:
            from_xml_tokens[qid].append([creation_date, text])
            nonskipped_sentences += len(text)
        # print(from_xml_tokens[qid])

        # print(from_xml_tokens[qid])
        # print("\n\n")

    print("done loading in xml dump")
    return from_xml_tokens, skipped_sentences, nonskipped_sentences


def construct_time_based_dataset(from_xml_tokens, labeled_data_file, out_base, label_set, start_id, sentence_ct,
                                 skipped_annotated):
    # Load in the full lines... store as a map from qid to a list of q's or a's, each of which is a list of lines
    lists_of_posts = defaultdict(list)
    qid = None
    lines = get_lines(labeled_data_file)
    for l in lines:
        if get_first(l).lower() in ['question_id', 'answer_to_question_id']:
            next(lines)
            qid = next(lines).strip().split("\t")[0]
            lists_of_posts[qid].append([])
        elif get_first(l).lower() == 'question_url':
            next(lines)
            next(lines)
        elif len(l.strip()) > 0:
            single_sent = [l]
            n = next(lines)
            while len(n.strip()) > 0:
                single_sent.append(n)
                n = next(lines)
            lists_of_posts[qid][-1].append(single_sent)

    print("done loading in annotated data")

    data_splits = [[], [], [], [], []]

    # determine date cutoffs for each split, print them out.
    # amount_per_split = sum([sum([len(text) for text in qid]) for qid in lists_of_posts])/5
    # all_create_dates = []
    # for qid in lists_of_posts:
    #     for text in lists_of_posts[qid]:
    #         create_date = get_create_date(qid, text, from_xml_tokens)
    #         if create_date is not None:
    #             all_create_dates.append(create_date)
    # amount_per_split = len(all_create_dates)//5
    # cutoffs = []
    # all_create_dates = sorted(all_create_dates)
    # for i in range(4):
    #     cutoffs.append(all_create_dates[(i+1)*amount_per_split])
    # cutoffs.append(datetime.datetime.strptime("2050-01-01", "%Y-%m-%d"))

    # print(cutoffs)

    # cutoffs determined using code above, for training data:
    cutoffs = [datetime.datetime(2012, 6, 27, 0, 0),
               datetime.datetime(2014, 3, 14, 0, 0),
               datetime.datetime(2015, 6, 28, 0, 0),
               datetime.datetime(2016, 10, 2, 0, 0),
               datetime.datetime(2050, 1, 1, 0, 0)]

    curr_id = start_id
    for qid in lists_of_posts:
        for text in lists_of_posts[qid]:
            # text is a list of lists
            # print("getting create date for qid %s with text %s\n" % (qid, str(text)))
            create_date = get_create_date(qid, text, from_xml_tokens)
            # print(create_date)
            # print("\n\n")
            if create_date is not None:
                for i in range(5):
                    if create_date < cutoffs[i]:
                        which_split = i
                        break

                json_elts, new_labels, curr_id = create_json_elts(text, create_date, qid, curr_id)
                data_splits[which_split].extend(json_elts)
                label_set |= new_labels
                sentence_ct += len(text)
            else:
                # print(text)
                skipped_annotated += len(text)

    # Go through lists_of_posts, matching them up with things we loaded in from the xml_dump files, then assign
    # to the correct dict... these can be directly written out as

    if labeled_data_file[-7:] == "dev.txt":
        for idx, split in enumerate(data_splits):
            with open(out_base + "_%d.json" % (idx + 1), 'r') as f:
                train_data_split = json.load(f)
            d_split = train_data_split + data_splits[idx]
            with open(out_base + "_%d.json" % (idx + 1), 'w') as f:
                json.dump(d_split, f)
    else:
        for idx, split in enumerate(data_splits):
            with open(out_base + "_%d.json" % (idx + 1), 'w') as f:
                json.dump(data_splits[idx], f)

    return curr_id, sentence_ct, skipped_annotated, label_set


def create_splits(entity_types_to_json_elts, entity_distribs, sorted_entities, total_size):
    splits = [[], [], [], [], []]
    selected_ids = set()

    ct = 0
    while sum([len(s) for s in splits]) < total_size:
        if ct % 1000 == 0:
            print(sum([len(s) for s in splits]))
            print(total_size)
        ct += 1
        for episode in range(5):
            distrib = entity_distribs[episode, :]
            # print(distrib)

            draw = np.where(np.random.multinomial(1, distrib))[0][0]
            # print(draw)
            entity_to_draw = sorted_entities[draw][0]

            while not len(entity_types_to_json_elts[entity_to_draw]):
                draw = np.where(np.random.multinomial(1, distrib))[0][0]
                # print(draw)
                entity_to_draw = sorted_entities[draw][0]

            def is_forbidden(s, ep):
                for t in s["tag_sequence"].split(" "):
                    if ("User_Interface_Element" in t and ep == 0) or \
                            ("Code_Block" in t and ep > 1) or \
                            ("Data_Structure" in t and ep < 3):
                        return True
                return s["id"] in selected_ids

            # Draw one of the samples from there, and remove it
            for sample in entity_types_to_json_elts[entity_to_draw]:
                # sample = entity_types_to_json_elts[entity_to_draw][i]
                if not is_forbidden(sample, episode):
                    splits[episode].append(sample)
                    entity_types_to_json_elts[entity_to_draw].remove(sample)
                    selected_ids.add(sample["id"])
                    break
    return splits


def load_data(labeled_data_file, lists_of_posts):
    qid = None
    lines = get_lines(labeled_data_file)
    for l in lines:
        if get_first(l).lower() in ['question_id', 'answer_to_question_id']:
            next(lines)
            qid = next(lines).strip().split("\t")[0]
            lists_of_posts[qid].append([])
        elif get_first(l).lower() == 'question_url':
            next(lines)
            next(lines)
        elif len(l.strip()) > 0:
            single_sent = [l]
            n = next(lines)
            while len(n.strip()) > 0:
                single_sent.append(n)
                n = next(lines)
            lists_of_posts[qid][-1].append(single_sent)

    print("done loading in annotated data of %s" % labeled_data_file)
    return lists_of_posts


def map_entity_types_to_json_elts(lists_of_posts, label_set, curr_id, sentence_ct):
    train_data = []
    # Create a map from entity types to sentences which are available
    entity_types_to_json_elts = defaultdict(list)

    for qid in lists_of_posts:
        for text in lists_of_posts[qid]:
            # text is a list of lists
            json_elts, new_labels, curr_id = create_json_elts(text, None, qid, curr_id)
            train_data.extend(json_elts)
            for local_new_labels, json_elt in zip(new_labels, json_elts):
                label_set |= local_new_labels
                for l in local_new_labels:
                    if l.startswith("B-"):
                        entity_types_to_json_elts[l[2:]].append(json_elt)
            sentence_ct += len(text)

    return train_data, entity_types_to_json_elts, label_set, curr_id, sentence_ct


def construct_skewed_dataset(labeled_data_files, out_dir, label_set, start_id, sentence_ct, skipped_annotated):
    lists_of_posts = defaultdict(list)
    lists_of_posts_test = defaultdict(list)

    # Load in the "train" and "dev" sets, which will be combined into a large train set
    for labeled_data_file in labeled_data_files:
        if labeled_data_file[-7:] == "dev.txt" or labeled_data_file[-9:] == "train.txt":
            lists_of_posts = load_data(labeled_data_file, lists_of_posts)
        else:
            lists_of_posts_test = load_data(labeled_data_file, lists_of_posts_test)

    curr_id = start_id
    train_data, entity_types_to_json_elts, \
    label_set, curr_id, sentence_ct = map_entity_types_to_json_elts(lists_of_posts, label_set, curr_id, sentence_ct)

    test_data, entity_types_to_json_elts_test, \
    label_set, curr_id, sentence_ct = map_entity_types_to_json_elts(lists_of_posts_test,
                                                                    label_set, curr_id, sentence_ct)

    # Get the initial distribution of the entity types
    # only do this for the training set, and then use the same distribution plus noise to sample test data
    entity_counts = count_entities(train_data)
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[0])

    # Sample the entity type distributions for each episode
    cts = np.array([e[1] for e in sorted_entities])
    normalization = np.sum(cts) / 5
    entity_distribs = np_gen.dirichlet(cts / normalization, 5)

    # Add class incrementality and drops of labels
    sorted_entity_labels = [e[0] for e in sorted_entities]
    user_interface_element_idx = sorted_entity_labels.index("User_Interface_Element")
    code_block_idx = sorted_entity_labels.index("Code_Block")
    data_structure_idx = sorted_entity_labels.index("Data_Structure")
    entity_distribs[0, user_interface_element_idx] = 0
    entity_distribs[0:3, data_structure_idx] = 0
    entity_distribs[2:, code_block_idx] = 0

    # For each episode, sample from the entity type distribution, and then pull a sentence with that entity type.
    # Cycle through the episodes so we can get as close to the intended distribution as possible.
    total_size = len(train_data) / 1.7
    train_splits = create_splits(entity_types_to_json_elts, entity_distribs, sorted_entities, total_size)

    # Save the train data
    for idx, split in enumerate(train_splits):
        with open(os.path.join(out_dir, "so_train_%d.json" % (idx + 1)), 'w') as f:
            json.dump(sorted(split, key=lambda x: int(x["id"])), f)

    # Save all the train data together, for running the baseline
    # Need to re-aggregate since we may have dropped some items from the training set during sampling episodes
    # Save 5 copies, so we can just pretend there are 5 training sets and 5 test sets for the code that
    # expects episodes.
    train_data = []
    for split in train_splits:
        train_data.extend(split)
    for idx in range(len(train_splits)):
        with open(os.path.join(out_dir, "so_train_all_%d.json" % (idx + 1)), 'w') as f:
            json.dump(sorted(train_data, key=lambda x: int(x["id"])), f)

    # Sample the test data according to a slightly perturbed version of the training data
    for ep in range(5):
        entity_distribs[ep, :] = np_gen.dirichlet(entity_distribs[ep, :]+1, 1)
    entity_distribs[0, user_interface_element_idx] = 0
    entity_distribs[0:3, data_structure_idx] = 0
    entity_distribs[2:, code_block_idx] = 0

    total_size = len(test_data) / 1.7
    test_splits = create_splits(entity_types_to_json_elts_test, entity_distribs, sorted_entities, total_size)

    for idx, split in enumerate(test_splits):
        with open(os.path.join(out_dir, "so_test_%d.json" % (idx + 1)), 'w') as f:
            json.dump(split, f)

    return curr_id, sentence_ct, skipped_annotated, label_set


def create_sample_tr_test_space(xml_dump_file):
    etree = ET.parse(xml_dump_file)
    root = etree.getroot()
    print(len(root))
    examples = []
    for row in root:
        print('next:')
        text = row.attrib["Body"]
        examples.append(text)
        print(tokenize(text))


def get_first(line):
    return line.strip().split("\t")[0]


def get_qids_of_interest(labeled_data_files, lists_of_posts_loc):
    lists_of_posts = defaultdict(list)
    qid = None
    for f in labeled_data_files:
        lines = get_lines(f)
        for l in lines:
            if get_first(l).lower() in ['question_id', 'answer_to_question_id']:
                next(lines)
                qid = next(lines).strip().split("\t")[0]
                lists_of_posts[qid].append([])
            elif get_first(l).lower() == 'question_url':
                next(lines)
                next(lines)
            elif len(l.strip()) > 0:
                single_sent = [get_first(l)]
                n = next(lines)
                while len(n.strip()) > 0:
                    single_sent.append(get_first(n))
                    n = next(lines)
                lists_of_posts[qid][-1].append(single_sent)
    with open(lists_of_posts_loc, "w") as f:
        json.dump(lists_of_posts, f)


def count_entities(data):
    entity_cts = Counter()
    for d in data:
        for t in d["tag_sequence"].strip().split(" "):
            if t.startswith("B-"):
                entity_cts[t[2:]] += 1
    return entity_cts


def get_entity_list(data_loc):
    entity_list = set()
    for episode in range(1, 6):
        # for train_test in ["train", "test"]:
        for train_test in ["train"]:
            with open(os.path.join(data_loc, "so_%s_%d.json" % (train_test, episode)), 'r') as f:
                data = json.load(f)
            entity_cts = count_entities(data)
            total = sum(entity_cts.values())
            for i in entity_cts:
                if float(entity_cts[i]) / total > 0.05:
                    entity_list.add(i)
    return sorted(list(entity_list))


def plot_entity_pie(entity_cts, episode, train_test, sorted_entities, entity_colors, out_base):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot()

    total = sum(entity_cts.values())
    # sorted_entities = sorted([i for i in entity_cts if float(entity_cts[i])/total > 0.01])
    # sorted_entities = sorted([i for i in entity_cts])

    sizes = [100 * float(entity_cts[i]) / total for i in sorted_entities]
    sorted_entities = [i for i in sorted_entities]
    sorted_entities.append("Other Entities")
    sizes.append(100 - sum(sizes))

    ax.pie(sizes, colors=[entity_colors[i] for i in sorted_entities], autopct='%1.1f%%', labels=sorted_entities)
    ax.axis('equal')

    plt.tight_layout(pad=15)
    train_test = "T" + train_test[1:]
    plt.title("%s Ep. %d" % (train_test, episode))
    name = "so_entity_pie_%d_%s" % (episode, train_test)
    plt.savefig("%s/plots/%s.png" % (out_base, name))
    plt.clf()


def plot_all_entity_pies(data_loc):
    entity_list = get_entity_list(data_loc)

    random.seed(15)
    all_colors = sorted(mcolors.CSS4_COLORS, key=lambda x: random.random())

    entity_colors = {entity_list[i]: all_colors[i] for i in range(len(entity_list))}
    entity_colors["Other Entities"] = all_colors[len(entity_list)]
    print("entity_colors")
    print(entity_colors)

    for episode in range(1, 6):
        for train_test in ["train", "test"]:
            with open(os.path.join(data_loc, "so_%s_%d.json" % (train_test, episode)), 'r') as f:
                data = json.load(f)
            entity_cts = count_entities(data)
            # sorted_entities = sorted(list(entity_cts.keys()))
            plot_entity_pie(entity_cts, episode, train_test, entity_list, entity_colors, data_loc)


def plot_average_percent_over_episodes(up_or_down, entities, train_test, entity_to_percentage_of_ep_1):
    avgs = [0.0] * 5
    for e in entities:
        for idx, p in enumerate(entity_to_percentage_of_ep_1[e]):
            avgs[idx] += p
    for i in range(5):
        avgs[i] /= len(entities)

    # make the plot
    plt.figure(figsize=(5, 4))
    ax = plt.subplot()

    plt.plot(range(1, 6), avgs)

    ax.set_xlim([0, 6])
    plt.xticks(range(1, 6))
    plt.title("Percent of ep 1 over eps (%s %s)" % (up_or_down, train_test))
    ax.set_ylabel("Average percent of ep 1")

    ax.legend()
    plt.tight_layout()
    name = "percent_of_ep_1_%s_%s" % (up_or_down, train_test)
    plt.savefig("/Users/juspayan/plots/%s.png" % name)


def plot_percentage_change_over_episodes():
    data_loc = "/Users/juspayan/Downloads"
    for train_test in ["train", "test"]:
        entity_percent_by_episode = defaultdict(list)
        for episode in range(1, 6):
            with open(os.path.join(data_loc, "so_%s_%d.json" % (train_test, episode)), 'r') as f:
                data = json.load(f)
            entity_cts = count_entities(data)
            total = sum(entity_cts.values())
            for e, c in entity_cts.items():
                entity_percent_by_episode[e].append(float(c) / total)

        entity_to_percentage_of_ep_1 = defaultdict(list)

        for e in entity_percent_by_episode.keys():
            percents = []
            for p in entity_percent_by_episode[e]:
                percents.append(100 * p / entity_percent_by_episode[e][0])
            entity_to_percentage_of_ep_1[e] = percents

        # Get the entities that change at least 20% up or down by the end, and plot those over time
        entities_up = set([e for e in entity_to_percentage_of_ep_1 if entity_to_percentage_of_ep_1[e][-1] > 120])
        entities_down = set([e for e in entity_to_percentage_of_ep_1 if entity_to_percentage_of_ep_1[e][-1] < 80])

        # entities_up = set([e for e in entity_to_percentage_of_ep_1 if entity_to_percentage_of_ep_1[e][-1] > 120 and
        #                    sum(entity_cts_by_episode[e]) > 50])
        # entities_down = set([e for e in entity_to_percentage_of_ep_1 if entity_to_percentage_of_ep_1[e][-1] < 80 and
        #                      sum(entity_cts_by_episode[e]) > 50])

        print("up: %s" % str(sorted(entities_up)))
        print(100 * float(sum([entity_percent_by_episode[e][0] for e in entities_up])) /
              float(sum([entity_percent_by_episode[e][0] for e in entity_percent_by_episode])))
        print("down: %s" % str(sorted(entities_down)))
        print(100 * float(sum([entity_percent_by_episode[e][0] for e in entities_down])) /
              float(sum([entity_percent_by_episode[e][0] for e in entity_percent_by_episode])))

        plot_average_percent_over_episodes("up", entities_up, train_test, entity_to_percentage_of_ep_1)
        plot_average_percent_over_episodes("down", entities_down, train_test, entity_to_percentage_of_ep_1)


def create_all_datasets(xml_dump_dir, data_dir, out_dir):
    labeled_data_files = ["train.txt", "dev.txt", "test.txt"]
    labeled_data_files = [os.path.join(data_dir, "StackOverflowNER/resources/annotated_ner_data/StackOverflow",
                                       f_name) for f_name in labeled_data_files]
    # get_qids_of_interest(labeled_data_files, lists_of_posts_loc)

    # from_xml_tokens, skipped_sentences, nonskipped_sentences = load_xml_dump(xml_dump_dir)

    curr_id = 0
    label_set = set()
    sentence_ct = 0
    skipped_annotated = 0

    # curr_id, sentence_ct, skipped_annotated, label_set = construct_time_based_dataset(from_xml_tokens,
    #                                                                                   labeled_data_file, out_base,
    #                                                                                   label_set, curr_id,
    #                                                                                   sentence_ct,
    #                                                                                   skipped_annotated)
    curr_id, sentence_ct, skipped_annotated, label_set = \
        construct_skewed_dataset(labeled_data_files, out_dir, label_set, curr_id, sentence_ct, skipped_annotated)

    print("Number of annotated sentences: %d" % sentence_ct)
    print("Number of annotated sentences skipped: %d" % skipped_annotated)
    # print("Number of sentences skipped in XML dump: %d" % skipped_sentences)
    # print("Number of sentences loaded in XML dump: %d" % nonskipped_sentences)

    label_map = {}
    for idx, l in enumerate(label_set):
        label_map[l] = idx

    with open(os.path.join(out_dir, "so_labels"), 'w') as labelf:
        json.dump(label_map, labelf)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_data_dir", default="~/Spring_2021", type=str)
    parser.add_argument("--xml_dump_dir", default="./so_data", type=str)
    parser.add_argument("--json_dir", default="./so_data", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(31415)
    # np.default_rng is seeded at top

    create_all_datasets(args.xml_dump_dir, args.labeled_data_dir, args.json_dir)
    plot_all_entity_pies(args.json_dir)
    # plot_percentage_change_over_episodes()
