import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, length=800):
    test_acc = get_all_results_for_one_algo(
        algorithm, dataset, goal, times, int(length))
    test_acc_data = np.average(test_acc, axis=0)
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, length=800):
    test_acc = np.zeros((times, length))
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + \
            algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc[i, :] = np.array(
            read_data_then_delete(file_name, delete=False))[:length]

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc

def print_par(args):
    
    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    # print("Client drop rate: {}".format(args.client_drop_rate))
    # print("Time select: {}".format(args.time_select))
    # print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)