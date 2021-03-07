''' This files considered results stored in a set of folders, each one containing
one csv for each seed used '''
import os
import csv
import re
import numpy as np
from matplotlib import pyplot as plt

stored_results_root = 'stored_results'
graphs_dir = 'graphs'

# tot episodes -1
n_episodes = 1000
tests_dict = dict()

stored_res_iterator = os.walk(stored_results_root)
# skip first level (test dir list)
next(stored_res_iterator)


def extract_values_from_csvs(folder, filename_list):
    seed_index = 0

    # remove non csv files
    pattern = re.compile('.*\\.csv$')
    filepath_list = [folder + os.sep + fn for fn in filename_list if bool(pattern.match(fn))]

    # one file for each seed
    values = np.zeros((len(filename_list), n_episodes))

    for fp in filepath_list:
        with open(fp, 'r') as file:
            file_reader = csv.reader(file)
            # skip header
            next(file_reader)
            # skip first episode
            next(file_reader)
            # keep only results column
            values[seed_index] = np.array(list(file_reader))[:, 1]
            seed_index += 1
    return values


def plot_single_result(test_name):
    plt.close()
    n_averaged_samples = 10
    values = tests_dict[test_name]
    for episode_series in values:
        averaged_episodes = np.mean(episode_series.reshape(-1, n_averaged_samples), axis=1)
        # 1000 episodes, n_averaged_samples = 10 -> x: 5, 15, 25, ... 995
        x = range(n_averaged_samples//2, n_episodes, n_averaged_samples)
        plt.plot(x, averaged_episodes)
    plt.title(test_name)
    plt.ylim([-600, 100])
    plt.xlim([0, 1000])
    graph_path = graphs_dir + os.sep + 'single_test' + os.sep + test_name + '.png'
    plt.savefig(graph_path)
    plt.cla()

if __name__ == '__main__':
    for dir_info in stored_res_iterator:
        dir_path = dir_info[0]
        dir_files = dir_info[2]
        # for each test we add a 2D matrix containing results for each seed
        # using test folder name as key
        key = dir_path.split(sep=os.sep)[-1]
        tests_dict[key] = extract_values_from_csvs(dir_path, dir_files)
        plot_single_result(key)
