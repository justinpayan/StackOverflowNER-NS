import argparse


def get_lines(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            yield line


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_file", type=str)
    parser.add_argument("--second_file", type=str)
    parser.add_argument("--third_file", type=str)
    parser.add_argument("--fourth_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    return args


def get_incorrect(lines):
    idx = 0
    curr = []
    incorrect_exs = {}
    correct = True

    for l in lines:
        if not l.strip():
            if not correct:
                incorrect_exs[idx] = curr
            idx += 1
            curr = []
            correct = True
        else:
            tok, ans, pred = l.strip().split(" ")
            curr.append(l)
            if ans != pred:
                correct = False
    return incorrect_exs


if __name__ == "__main__":
    args = parse_args()
    data_path_first = args.first_file
    data_path_second = args.second_file
    data_path_third = args.third_file
    data_path_fourth = args.fourth_file
    out_path = args.output_file

    # Get examples that degrade between the first and second but not the third and fourth.

    lines_first = get_lines(data_path_first)
    lines_second = get_lines(data_path_second)

    lines_third = get_lines(data_path_third)
    lines_fourth = get_lines(data_path_fourth)

    wrong_first = get_incorrect(lines_first)
    wrong_second = get_incorrect(lines_second)
    wrong_third = get_incorrect(lines_third)
    wrong_fourth = get_incorrect(lines_fourth)

    first_degradations = set(wrong_second.keys()) - set(wrong_first.keys())
    second_degradations = set(wrong_fourth.keys()) - set(wrong_third.keys())

    with open(out_path, 'w') as f:
        for idx in sorted(first_degradations - second_degradations):
            f.write(''.join(wrong_second[idx]) + "\n")
