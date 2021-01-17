import json, os, re
import numpy as np
import matplotlib.pyplot as plt


def get_f1_scores(model_dir, test_episode, gt_label_cts=None):
    print(test_episode)
    try:
        conll_score_out_file = os.path.join(model_dir, "ner_conll_score_out_so_{}_{}".format(test_episode, "finish"))
        f1_scores = {}
        with open(conll_score_out_file, 'r') as f:
            for l in f.readlines()[2:]:
                label = re.search("\s*(\S*):\s*", l).group(1)
                if gt_label_cts is not None:
                    if label in gt_label_cts and gt_label_cts[label] > 20:
                        f1_score = float(re.search("\s*FB1:\s*(\S*)\s*", l).group(1))
                        f1_scores[label] = f1_score
                else:
                    f1_score = float(re.search("\s*FB1:\s*(\S*)\s*", l).group(1))
                    f1_scores[label] = f1_score

        print(f1_scores)
        return f1_scores
    except IOError:
        return None


def compute_entity_f1(f1_scores):
    return np.mean([f1_scores[i] for i in f1_scores if not i.endswith("Intent")])


def make_ner_plot(test_episode, entity_f1s):
    plt.figure(figsize=(5, 4))
    ax = plt.subplot()

    for method, scores in entity_f1s.items():
        x = list(range(1, 6))
        plt.plot(x, scores, label=method)

    ax.set_xlim([x[0] - .1, x[-1] + .1])
    plt.xticks(x)
    ax.set_xticklabels(x)
    ax.set_ylim([40, 60])
    plt.title("Macro F1 on Episode %d Over Time" % test_episode)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("# Episodes Trained")

    ax.legend()
    plt.tight_layout()
    name = "ner_so_ep_%d_improvedgen_02" % test_episode
    plt.savefig("plots/%s.png" % name)


def make_ner_plot_indiv(entity_f1s):
    plt.figure(figsize=(5, 4))
    ax = plt.subplot()

    for method, scores in entity_f1s.items():
        x = list(range(1, 6))
        plt.plot(x, scores, label=method)

    ax.set_xlim([x[0] - .1, x[-1] + .1])
    plt.xticks(x)
    ax.set_xticklabels(x)
    ax.set_ylim([40, 60])
    plt.title("Macro F1 on Each Episode")
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Train/Test Episode")

    ax.legend()
    plt.tight_layout()
    name = "ner_so_all_eps_indiv_improvedgen_02"
    plt.savefig("plots/%s.png" % name)


def plot_over_time():
    # Compute macro f1 for every tag. We can print the tables for intents and NER separately at the end.

    ft_dir = "models/gpt2/finetune/so_1_so_2_so_3_so_4_so_5_tasksner"
    gen_dir = "models/gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_tasksner_improvedgen"
    real_dir = "models/gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_tasksner"

    base_dirs = {"No Replay": ft_dir, "Gen (Proactive)": gen_dir, "Real Replay": real_dir}

    final_avg_f1s = {"No Replay": [], "Gen (Proactive)": [], "Real Replay": []}

    for test_episode in range(1, 6):
        entity_f1s = {"No Replay": [], "Gen (Proactive)": [], "Real Replay": []}

        # with open("data/music-train-data/label_counts_%d_test" % test_episode) as f:
        #     gt_label_cts = json.load(f)

        for method in base_dirs:
            for train_episode in range(1, 6):
                model_dir = os.path.join(base_dirs[method], "so_%d" % train_episode)
                f1_scores = get_f1_scores(model_dir, test_episode)
                if f1_scores:
                    f1 = compute_entity_f1(f1_scores)
                    entity_f1s[method].append(f1)
                    if train_episode == 5:
                        final_avg_f1s[method].append(f1)
        # Now we can make plots for this ROW of the plots. 1 test episode over all training episodes.
        print(test_episode)
        make_ner_plot(test_episode, entity_f1s)

    print(final_avg_f1s)
    print()
    for method in final_avg_f1s:
        print("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (method,
                                                                     final_avg_f1s[method][0],
                                                                     final_avg_f1s[method][1],
                                                                     final_avg_f1s[method][2],
                                                                     final_avg_f1s[method][3],
                                                                     final_avg_f1s[method][4],
                                                                     float(np.mean(final_avg_f1s[method]))))


def plot_indiv_test_set_perfs():
    ft_dir = "models/gpt2/finetune/so_1_so_2_so_3_so_4_so_5_tasksner"
    gen_dir = "models/gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_tasksner_improvedgen"
    real_dir = "models/gpt2/lll/so_1_so_2_so_3_so_4_so_5_0.2_0.25_6.25e-05_real_tasksner"

    base_dirs = {"No Replay": ft_dir, "Gen (Proactive)": gen_dir, "Real Replay": real_dir}

    entity_f1s = {"No Replay": [], "Gen (Proactive)": [], "Real Replay": []}

    for episode in range(1, 6):

        for method in base_dirs:
            model_dir = os.path.join(base_dirs[method], "so_%d" % episode)
            f1_scores = get_f1_scores(model_dir, episode)
            if f1_scores:
                f1 = compute_entity_f1(f1_scores)
                entity_f1s[method].append(f1)
    make_ner_plot_indiv(entity_f1s)


if __name__ == "__main__":
    plot_over_time()
    plot_indiv_test_set_perfs()






