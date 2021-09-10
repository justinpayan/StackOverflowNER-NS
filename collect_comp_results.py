import argparse
import os
import re

import json
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter


all_types = ["accuracy", "Algorithm", "Application", "Class_Name", "Code_Block", "Data_Structure", "Data_Type",
             "Device", "Error_Name", "File_Name",
             "File_Type", "Function_Name", "HTML_XML_Tag", "Keyboard_IP", "Language", "Library", "Library_Class",
             "Library_Function", "Library_Variable", "License", "Operating_System", "Organization", "Output_Block",
             "User_Interface_Element", "User_Name", "Value", "Variable_Name",
             "Version", "Website"]


def list_results(file_name):
    results = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # for which_result in ["accuracy", "Application", "Code_Block", "Data_Structure", "Library_Class",
    #                      "User_Interface_Element", "Variable_Name"]:
    for which_result in all_types:
        fscore = None
        for l in lines:
            l = l.strip()
            if l.startswith(which_result + ":"):
                fscore = float(re.search("FB1:\s*(\d+\.\d+)", l).group(1))
                results.append(fscore)
        if fscore is None:
            results.append(0.0)
    return results


def get_results_one_setting(results_location):
    results = []

    ep_5_dir = None
    for ep_dir in os.listdir(results_location):
        if ep_dir.endswith("_5"):
            ep_5_dir = os.path.join(results_location, ep_dir)

    for test_ep in range(1, 6):
        for result_file in os.listdir(ep_5_dir):
            if result_file.endswith("%d_finish" % test_ep) and result_file.startswith("ner_conll_score_out"):
                rf_full = os.path.join(ep_5_dir, result_file)
                results.append(list_results(rf_full))

    results = np.array(results)
    return np.mean(results, axis=0).tolist()


if __name__ == "__main__":

    # Collect comprehensive results for temporal
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/finetune/so_t_all_1_so_t_all_2_so_t_all_3_so_t_all_4_so_t_all_5"
    temp_baseline = get_results_one_setting(results_location)
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/finetune/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5"
    temp_no_replay = get_results_one_setting(results_location)
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/lll/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_0.2_0.25_6.25e-05_real_improvedgen"
    temp_replay = get_results_one_setting(results_location)

    # Collect comprehensive results for skewed
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/finetune/so_all_1_so_all_2_so_all_3_so_all_4_so_all_5"
    skew_baseline = get_results_one_setting(results_location)
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/finetune/so_1_so_2_so_3_so_4_so_5"
    skew_no_replay = get_results_one_setting(results_location)
    results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_improvedgen"
    skew_replay = get_results_one_setting(results_location)

    temporal_type_cts = Counter()
    for i in range(1, 6):
        with open("/home/jpayan/Lamolrelease/so_data/so_temporal_test_%d.json" % i) as f:
            testset = json.load(f)
            for ex in testset:
                for tag in ex["tag_sequence"].strip().split():
                    if tag.startswith("B"):
                        temporal_type_cts[tag[2:]] += 1
    temp_cts_list = []
    for t in all_types:
        temp_cts_list.append(temporal_type_cts[t]/5)
    temp_cts_list.insert(0, sum(temp_cts_list))

    skew_type_cts = Counter()
    for i in range(1, 6):
        with open("/home/jpayan/Lamolrelease/so_data/so_test_%d.json" % i) as f:
            testset = json.load(f)
            for ex in testset:
                for tag in ex["tag_sequence"].strip().split():
                    if tag.startswith("B"):
                        skew_type_cts[tag[2:]] += 1
    skew_cts_list = []
    for t in all_types:
        skew_cts_list.append(skew_type_cts[t]/5)
    skew_cts_list.insert(0, sum(skew_cts_list))

    gdumb_results = {}
    k = 1500
    results_k = []
    for seed in range(10):
        gdumb_str = "_".join(["gdumb_%d_%d_%d" % (k, seed, i) for i in range(1, 6)])
        results_location = "/mnt/nfs/scratch1/jpayan/Lamolrelease/models/2021-09-08-21-05-03/gpt2/finetune/" + gdumb_str
        results_k.append(get_results_one_setting(results_location))
    results_k = np.array(results_k)
    gdumb_means = np.mean(results_k, axis=0)
    gdumb_stds = np.std(results_k, axis=0)

    temp_results_table = np.array([temp_baseline, temp_no_replay, temp_replay]).transpose()
    skew_results_table = np.array([skew_baseline, skew_no_replay, skew_replay]).transpose()

    def map_type(ent_type):
        if ent_type == "accuracy":
            return "Overall"
        else:
            return re.sub("_", "", ent_type)

    for i in range(len(all_types)):
        to_print = "\\small " + map_type(all_types[i])
        to_print += " & " + " & ".join(["%.2f" % x for x in temp_results_table[i, :]])
        to_print += (" & %.2f" % temp_cts_list[i])
        to_print += " \\\\"
        print(to_print)

    for i in range(len(all_types)):
        to_print = "\\small " + map_type(all_types[i])
        to_print += " & " + " & ".join(["%.2f" % x for x in skew_results_table[i, :]])
        to_print += " & $%.2f \\pm %.2f$" % (gdumb_means[i], gdumb_stds[i])
        to_print += (" & %.2f" % skew_cts_list[i])
        to_print += " \\\\"
        print(to_print)


