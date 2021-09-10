import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/mnt/nfs/scratch1/jpayan/Lamolrelease/models", type=str)
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


def get_overall_f1(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for l in lines:
        l = l.strip()
        if l.startswith("accuracy"):
            fscore = float(re.search("FB1:\s*(\d+\.\d+)", l).group(1))
            return fscore

    return 0.0


def get_results_ep_1(results_location):
    results = []

    ep_5_dir = None
    for ep_dir in sorted(os.listdir(results_location)):
        if not ep_dir.startswith("log"):
            ep_dir = os.path.join(results_location, ep_dir)
            for result_file in sorted(os.listdir(ep_dir)):
                if result_file.endswith("1_finish") and result_file.startswith("ner_conll_score_out"):
                    rf_full = os.path.join(ep_dir, result_file)
                    results.append(get_overall_f1(rf_full))

    print(results_location)
    print(results)
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


def get_results_one_setting_agg_over_time(results_location):
    results = []

    ep_5_dir = None
    for ep_dir in os.listdir(results_location):
        if ep_dir.endswith("_5"):
            ep_5_dir = os.path.join(results_location, ep_dir)

    for test_ep in range(1, 6):
        for result_file in os.listdir(ep_5_dir):
            if result_file.endswith("%d_finish" % test_ep) and result_file.startswith("ner_conll_score_out"):
                rf_full = os.path.join(ep_5_dir, result_file)
                results.append(get_overall_f1(rf_full))

    return results


def collect_results(results_dir):
    results_ep_5_train_all_eps_test = {"Temporal": {"Baseline": [], "No Replay": [], "Real Replay": []},
                                       "Skewed": {"Baseline": [], "No Replay": [], "Real Replay": []}}

    results_ep_5_train_all_eps_test_agg_over_time = {"Temporal": {"Baseline": [], "No Replay": [], "Real Replay": []},
                                                     "Skewed": {"Baseline": [], "No Replay": [], "Real Replay": []},
                                                     }

    results_all_eps_train_ep_1_test = {"Temporal": {"No Replay": [], "Real Replay": []},
                                       "Skewed": {"No Replay": [], "Real Replay": []}}

    # subdirs = {"Temporal":
    #                {"Baseline": "finetune/so_t_all_1_so_t_all_2_so_t_all_3_so_t_all_4_so_t_all_5_tasksner",
    #                "No Replay": "finetune/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_tasksner",
    #                "Real Replay":
    #                "lll/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_0.2_0.25_6.25e-05_real_tasksner_improvedgen"},
    #            "Skewed":
    #                {"Baseline": "finetune/so_all_1_so_all_2_so_all_3_so_all_4_so_all_5_tasksner",
    #                 "No Replay": "finetune/so_1_so_2_so_3_so_4_so_5_tasksner",
    #                 "Real Replay": "lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_tasksner_improvedgen"}}

    subdirs = {"Temporal":
                   {"Baseline": "gpt2/finetune/so_t_all_1_so_t_all_2_so_t_all_3_so_t_all_4_so_t_all_5",
                    "No Replay": "gpt2/finetune/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5",
                    "Real Replay": "gpt2/lll/so_t_1_so_t_2_so_t_3_so_t_4_so_t_5_0.2_0.25_6.25e-05_real_improvedgen"},
               "Skewed":
                   {"Baseline": "gpt2/finetune/so_all_1_so_all_2_so_all_3_so_all_4_so_all_5",
                    "No Replay": "gpt2/finetune/so_1_so_2_so_3_so_4_so_5",
                    "Real Replay": "gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_improvedgen"}
               }

    for ep_type in ["Temporal", "Skewed"]:
        for train_setting in ["Baseline", "No Replay", "Real Replay"]:
            results_location = os.path.join(results_dir, subdirs[ep_type][train_setting])
            results_ep_5_train_all_eps_test[ep_type][train_setting] = get_results_one_setting(results_location)
            results_ep_5_train_all_eps_test_agg_over_time[ep_type][train_setting] = \
                get_results_one_setting_agg_over_time(results_location)
            # if train_setting != "Baseline":
            #     results_all_eps_train_ep_1_test[ep_type][train_setting] = get_results_ep_1(results_location)

    return results_ep_5_train_all_eps_test, results_ep_5_train_all_eps_test_agg_over_time
    # return results_ep_5_train_all_eps_test, results_all_eps_train_ep_1_test, results_ep_5_train_all_eps_test_agg_over_time


def print_table(r):
    for ep_type in ["Temporal", "Skewed_1", "Skewed_5", "Skewed_10"]:
        print("& Baseline & " + " & ".join(["%.3f" % i for i in r[ep_type]["Baseline"]]) + "\\\\")
        print(ep_type + " & No Replay & " + " & ".join(["%.3f" % i for i in r[ep_type]["No Replay"]]) + "\\\\")
        print("& Real Replay & " + " & ".join(["%.3f" % i for i in r[ep_type]["Real Replay"]]) + "\\\\")
        print("\\midrule")


# def make_plots(r):
#     markers = {"Temporal": "o", "Skewed": "x"}
#     linestyles = {"No Replay": "--", "Real Replay": "-"}
#     for ep_type in ["Temporal", "Skewed"]:
#         for train_setting in ["No Replay", "Real Replay"]:
#             plt.plot(r[ep_type][train_setting],
#                      marker=markers[ep_type],
#                      linestyle=linestyles[train_setting],
#                      color="k",
#                      label="%s - %s" % (ep_type, train_setting))
#     plt.xlabel("Episode", fontsize="medium")
#     plt.xticks(ticks=range(5), labels=[str(i) for i in range(1, 6)])
#     plt.ylabel("Episode 1 Test F1", fontsize="medium")
#     plt.legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
#     plt.savefig("ep_1_time.png")


def make_plots_all_tests(r, on_train, ep_type):
    linestyles = {"No Replay": "--", "Real Replay": "-", "Baseline": ":"}
    relabels = {"No Replay": "CL without Replay", "Real Replay": "CL with Real Replay", "Baseline": "Baseline (non-CL)"}
    color_map = {"No Replay": "royalblue", "Real Replay": "darkorange", "Baseline": "forestgreen"}
    plt.clf()
    for train_setting in ["Baseline", "Real Replay", "No Replay"]:
        plt.plot(r[ep_type][train_setting],
                 linestyle=linestyles[train_setting],
                 color=color_map[train_setting],
                 marker='o',
                 label=relabels[train_setting])
        plt.ylim(ymin=38,ymax=102)
    if on_train:
        plt.xlabel("Train Episode", fontsize="large")
        plt.xticks(ticks=range(5), labels=[str(i) for i in range(1, 6)])
        plt.ylabel("Train F1", fontsize="large")
        plt.legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
        plt.savefig("%s_tests_on_train_over_time.png" % ep_type)
    else:
        plt.xlabel("Test Episode", fontsize="large")
        plt.xticks(ticks=range(5), labels=[str(i) for i in range(1, 6)])
        plt.ylabel("Test F1", fontsize="large")
        plt.legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
        plt.savefig("%s_tests_over_time.png" % ep_type)


if __name__ == "__main__":
    args = parse_args()

    # all_results, ep_1_over_time, test_on_train_all_eps = collect_results(args.results_dir)
    all_results, test_on_train_all_eps = collect_results(args.results_dir)

    # print_table(all_results)
    # make_plots(ep_1_over_time)
    # print(test_on_train_all_eps)
    # make_plots_all_tests(test_on_train_all_eps, True, "Skewed_1")
    make_plots_all_tests(test_on_train_all_eps, False, "Skewed")
    make_plots_all_tests(test_on_train_all_eps, False, "Temporal")

    # test_all_eps = {'Temporal': {'Baseline': [52.22, 54.12, 50.75, 48.06, 53.16],
    #                              'No Replay': [52.77, 55.69, 49.85, 47.94, 50.39],
    #                              'Real Replay': [51.17, 52.23, 50.6, 44.27, 49.48]},
    #                 'Skewed': {'Baseline': [45.81, 48.08, 51.29, 57.1, 56.56],
    #                            'No Replay': [30.59, 43.92, 45.57, 53.39, 51.3],
    #                            'Real Replay': [43.52, 45.26, 46.01, 57.31, 52.63]}}
    # make_plots_all_tests(test_all_eps, False)
