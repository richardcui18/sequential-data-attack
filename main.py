import uniform_attack
import numpy as np
import sys
from itertools import product
import hmm_rl
from options import Options
import datasets
sys.stdout.reconfigure(line_buffering=True)

# helper function to generate all possible subsets of size 1 given a PR state
def generate_subsets_from_pr(state):
    result = []
    for elements in product(*state):
        result.append([[el] for el in elements])
    return result


# helper function to determine whether a TL state is a subset of a PR state
def is_subset(tl_state, pr_state):
    for dim1, dim2 in zip(tl_state, pr_state):
        if not set(dim1).issubset(set(dim2)):
            return False
    return True

def generate_possible_tl_states_from_pr_states(pr_true_sequence_with_feature_names):
    # calculate PR sizes
    pr_sizes = []
    for pr in pr_true_sequence_with_feature_names:
        cur_size = 1
        for pr_dim in pr:
            cur_size *= len(pr_dim)
        pr_sizes.append(cur_size)

    # calculate possible TL states
    possible_tl_states = []
    for i in range(len(pr_true_sequence_with_feature_names)):
        possible_tl_states += generate_subsets_from_pr(pr_true_sequence_with_feature_names[i])

    for tl_state in possible_tl_states.copy():
        # Count the number of observed PR states that this TL state is a subset of
        subset_count = 0
        for pr_state in pr_true_sequence_with_feature_names:
            if is_subset(tl_state, pr_state):
                subset_count += 1
        # remove TL states that are not a subset of any observed PR states
        if subset_count == 0:
            possible_tl_states.remove(tl_state)
    
    return possible_tl_states


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    seed_attacker = 10

    # Hyperparameters (for testing)
    delta = args.delta
    num_iter = args.num_iter

    # Get dataset
    dataset = datasets.get_dataset(args.dataset, args.deviation_amount_user, args.lambda_value)


    hmm_and_baseline_comparison = {
        'Trajectory': [],
        'HMM Average Error (m)': [],
        'Baseline Average Error (m)': [],
        'HMM Max Error (m)': [],
        'Baseline Max Error (m)': [],
        'Baseline Average Error - HMM Average Error (m)': [],
        'Trajectory Length': dataset['total_times']
    }

    tl_hat_hmm_trajectories = []
    tl_hat_baseline_trajectories = []

    # Attacking each trajectory
    for i in range(len(dataset['tl_true_sequences'])):
        print()
        print('****************** Trajectory ' + str(i+1) + ' of ' + str(len(dataset['tl_true_sequences'])) + ' ******************')
        print()
        tl_true_sequence = dataset['tl_true_sequences'][i]
        tl_true_sequence_with_feature_names = dataset['tl_true_sequences_with_feature_names'][i]
        pr_true_sequence_with_feature_names = dataset['pr_true_sequences_with_feature_names'][i]
        total_time = dataset['total_times'][i]

        # Generate possible TL states from given PR states
        possible_tl_states = generate_possible_tl_states_from_pr_states(pr_true_sequence_with_feature_names)

        # Run HMM-RL algorithm
        results_weighted = hmm_rl.run_rl_algorithm(pr_true_sequence_with_feature_names, possible_tl_states, 
                        args.lambda_value, total_time, delta, num_iter, dataset['unique_values_on_each_dimension'], 
                        tl_true_sequence_with_feature_names, seed_attacker, dataset['cut_num_to_lon'], dataset['cut_num_to_lat'], args.k, args.gamma)

        # Calculate error
        euclidean_distance_list = []
        for day in range(len(tl_true_sequence_with_feature_names)):
            euclidean_distance_list.append(hmm_rl.calc_euclidean_distance(results_weighted[-1][day], tl_true_sequence_with_feature_names[day], dataset['cut_num_to_lon'], dataset['cut_num_to_lat']))
        mean_euclidean_distance = np.mean(np.array(euclidean_distance_list))

        # Run baseline algorithm
        euclidean_distance_list_bl, mean_euclidean_distance_bl, predicted_tl_bl, largest_deviation_baseline = uniform_attack.pr_uniform_attack(total_time, pr_true_sequence_with_feature_names, tl_true_sequence_with_feature_names, dataset['cut_num_to_lon'], dataset['cut_num_to_lat'])

        print(f'HMM-RL average Euclidean distance for trajectory {str(i+1)} (m):', mean_euclidean_distance)
        print(f'Baseline model mean Euclidean distance for trajectory {str(i+1)} (m):', mean_euclidean_distance_bl)
        print(f'HMM-RL max Euclidean distance for trajectory {str(i+1)} (m):', results_weighted[-4])
        print(f'Baseline model max Euclidean distance for trajectory {str(i+1)} (m):', largest_deviation_baseline)

        tl_hat_hmm_trajectories.append(results_weighted[-1])
        tl_hat_baseline_trajectories.append(predicted_tl_bl)

        hmm_and_baseline_comparison['Trajectory'].append(str(i+1))
        hmm_and_baseline_comparison['HMM Average Error (m)'].append(mean_euclidean_distance)
        hmm_and_baseline_comparison['Baseline Average Error (m)'].append(mean_euclidean_distance_bl)
        hmm_and_baseline_comparison['HMM Max Error (m)'].append(results_weighted[-4])
        hmm_and_baseline_comparison['Baseline Max Error (m)'].append(largest_deviation_baseline)
        hmm_and_baseline_comparison['Baseline Average Error - HMM Average Error (m)'].append(mean_euclidean_distance_bl-mean_euclidean_distance)

    print()
    print("---------- Aggregate Results Across All Trajectories ----------")
    print('Average HMM-RL Euclidean distance for deviation '+str(args.deviation_amount_user)+' : ' + str(np.mean(np.array(hmm_and_baseline_comparison['HMM Average Error (m)']))))
    print('Average Baseline Euclidean distance for deviation '+str(args.deviation_amount_user)+' : ' + str(np.mean(np.array(hmm_and_baseline_comparison['Baseline Average Error (m)']))))
    print('Max HMM-RL Euclidean distance for deviation '+str(args.deviation_amount_user)+' : ' + str(np.mean(hmm_and_baseline_comparison['HMM Max Error (m)'])))
    print('Max Baseline Euclidean distance for deviation '+str(args.deviation_amount_user)+' : ' + str(np.mean(hmm_and_baseline_comparison['Baseline Max Error (m)'])))
