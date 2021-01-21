import argparse
import os
import re

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/mnt/nfs/scratch1/jpayan/Lamolrelease/models/gpt2", type=str)
    args = parser.parse_args()
    return args


def list_results(file_name):
    results = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for which_result in ["accuracy", "Application", "Code_Block", "Data_Structure", "Library_Class",
                         "User_Interface_Element", "Variable_Name"]:
        fscore = None
        for l in lines:
            l = l.strip()
            if l.startswith(which_result):
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


def collect_results(results_dir):
    results = {"Temporal": {"Baseline": [], "No Replay": [], "Real Replay": []},
               "Skewed": {"Baseline": [], "No Replay": [], "Real Replay": []}}

    subdirs = {"Temporal":
                   {"Baseline": "finetune/so_t_all_1_so_t_all_2_so_t_all_3_so_t_all_4_so_t_all_5_tasksner",
                   "No Replay": "finetune/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_tasksner",
                   "Real Replay": "lll/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_0.2_0.25_6.25e-05_real_tasksner_improvedgen"},
               "Skewed":
                   {"Baseline": "finetune/so_all_1_so_all_2_so_all_3_so_all_4_so_all_5_tasksner",
                    "No Replay": "finetune/so_1_so_2_so_3_so_4_so_5_tasksner",
                    "Real Replay": "lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_tasksner_improvedgen"}}

    for ep_type in ["Temporal", "Skewed"]:
        for train_setting in ["Baseline", "No Replay", "Real Replay"]:
            results_location = os.path.join(results_dir, subdirs[ep_type][train_setting])
            results[ep_type][train_setting] = get_results_one_setting(results_location)

    return results


def print_table(r):
    for ep_type in ["Temporal", "Skewed"]:
        print("& Baseline & " + " & ".join(["%.3f" % i for i in r[ep_type]["Baseline"]]) + "\\\\")
        print(ep_type + " & No Replay & " + " & ".join(["%.3f" % i for i in r[ep_type]["No Replay"]]) + "\\\\")
        print("& Real Replay & " + " & ".join(["%.3f" % i for i in r[ep_type]["Real Replay"]]) + "\\\\")


if __name__ == "__main__":
    args = parse_args()

    all_results = collect_results(args.results_dir)
    print_table(all_results)
