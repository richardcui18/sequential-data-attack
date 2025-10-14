import itertools
import numpy as np
from hmmlearn import hmm
import math
import random
from sklearn.preprocessing import normalize
import uniform_attack
from itertools import product, combinations
import geopy.distance
import sys
import os
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)

pass_num = 0

# Helper function: find all grids that are within a given PR
def generate_grids_from_pr(published_region):
    combs = list(itertools.product(*published_region))
    tl_list = [list(map(lambda x: [x], combination)) for combination in combs]
    
    return tl_list

# Helper function to determine whether a TL state is a subset of a PR state
def is_subset(tl_state, pr_state):
    # num_dim = len(tl_state)
    for num_dim in range(len(tl_state)):
        if not set(tl_state[num_dim]) <= (set(pr_state[num_dim])):
            return False
    
    return True

# Helper function to get PR index from state
def get_pr_index(pr_state, pr_state_dict):
    for key in pr_state_dict.keys():
        if pr_state_dict[key] == pr_state:
            return key
    return -1

def generate_supersets_from_tl(state, min_size, max_size, x_bound, y_bound):
    x = int(state[0][0])
    y = int(state[1][0])

    def get_decompositions(size):
        """
        Find all pairs (n, k) such that n * k = size.
        This includes both (n, k) and (k, n) if n != k.
        """
        decompositions = []
        for n in range(1, int(size ** 0.5) + 1):
            if size % n == 0:
                k = size // n
                decompositions.append((n, k))
                if n != k:
                    decompositions.append((k, n))  # Add commutative pair
        return decompositions

    def generate_sequences(center, length, bound):
        """
        Generate all possible sequences around a center point of given length within bounds.
        For example, for center = 5, length = 3, and bound = 10:
        - [3, 4, 5] or [4, 5, 6], but all values must be within [1, 10]
        """
        sequences = []
        for i in range(length):
            start = max(center - i, bound[0])  # Ensure the start doesn't go below the lower bound
            end = start + length - 1
            if end > bound[1]:  # Ensure the end doesn't exceed the upper bound
                continue
            sequences.append([str(j) for j in range(start, start + length)])
        return sequences

    result = []

    # Loop through all sizes in the range
    for size in range(min_size, max_size + 1):
        decompositions = get_decompositions(size)  # Get (n, k) pairs

        for n, k in decompositions:
            # Generate all possible sequences for x based on n, ensuring bounds
            x_sequences = generate_sequences(x, n, x_bound)

            # Generate all possible sequences for y based on k, ensuring bounds
            y_sequences = generate_sequences(y, k, y_bound)

            # Take the Cartesian product of x and y sequences to form grids
            for x_seq, y_seq in product(x_sequences, y_sequences):
                result.append([x_seq, y_seq])

    return result

# Define a function to suppress stdout and stderr
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def run_rl_algorithm(pr_true_sequence_with_feature_names, possible_tl_states, lambda_value, total_time, delta,
                     num_iter, unique_values_on_each_dimension, tl_true_sequence_with_feature_names, seed,
                     cut_num_to_lon, cut_num_to_lat, window_size, gamma_attacker):
    # initialization
    observed_sequence = []
    pr_state_dict = {}
    pr_counter = 0
    tl_state_dict = {}
    tl_counter = 0

    pr_union = unique_values_on_each_dimension
    min_pr_size = math.floor(1 / lambda_value)
    pr_union_size = 1
    for dim in pr_union:
        pr_union_size *= len(dim)
    
    random.seed(seed)
    possible_pr_states = []
    x_range_int = list(map(int, unique_values_on_each_dimension[0]))
    y_range_int = list(map(int, unique_values_on_each_dimension[1]))
    for tl_state in possible_tl_states:
        possible_pr_states += generate_supersets_from_tl(tl_state, min_pr_size, min_pr_size+gamma_attacker, (min(x_range_int), max(x_range_int)), (min(y_range_int), max(y_range_int)))

    possible_pr_states = possible_pr_states + [pr for pr in pr_true_sequence_with_feature_names if pr not in possible_pr_states]
    
    # building TL index to state dict
    for tl_state in possible_tl_states:
        if tl_state not in tl_state_dict.values():
            tl_state_dict[tl_counter] = tl_state
            tl_counter += 1

    # check if all PR states is a superset of some TL state
    for pr_state in possible_pr_states:
        subset_num = 0
        for tl_state in possible_tl_states:
            if is_subset(tl_state, pr_state):
                subset_num += 1
        if subset_num == 0:
            possible_pr_states.remove(pr_state)

    # build PR index to state dict
    for pr in possible_pr_states:
        if pr not in pr_state_dict.values():
            pr_state_dict[pr_counter] = pr
            observed_sequence.append(pr_counter)
            pr_counter += 1

    observed_sequence = []
    for pr_observed_state in pr_true_sequence_with_feature_names:
        observed_sequence.append(get_pr_index(pr_observed_state, pr_state_dict))

    observed_sequence = np.array(observed_sequence)

    ## Initialize for Baum-Welch
    num_hidden_states = len(tl_state_dict)
    num_observed_states = len(pr_state_dict)
    
    transition_prior = np.ones((num_hidden_states, num_hidden_states))

    hidden_states = range(num_hidden_states)
    observed_states = range(num_observed_states)
    immutable_index = list(itertools.product(hidden_states, observed_states))
    immutable_index = []
    emission_prior = np.ones((num_hidden_states, num_observed_states))
    for tl_index, tl_state in tl_state_dict.items():
        for pr_index, pr_state in pr_state_dict.items():
            if is_subset(tl_state, pr_state):
                emission_prior[tl_index][pr_index] = 1
            else:
                immutable_index.append((tl_index, pr_index))
    

    initial_distribution_prior = np.ones(num_hidden_states)
    
    # start of RL-based algorithm

    transition_prob_matrix_forward = transition_prior.copy()
    transition_prob_matrix_backward = transition_prior.copy()
    emission_prob_matrix = emission_prior.copy()

    direction = 'backward'

    transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, model, tl_prediction_for_this_pass, _ = one_pass(
        transition_prob_matrix_forward, emission_prob_matrix, initial_distribution_prior, total_time, delta, lambda_value, 
        observed_sequence, pr_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 tl_state_dict, pr_state_dict, tl_true_sequence_with_feature_names,
             unique_values_on_each_dimension, cut_num_to_lon, cut_num_to_lat, gamma_attacker)
    
    euclidean_distance_day_pass = []
    euclidean_distance_list = []
    for day in range(len(tl_true_sequence_with_feature_names)):
        euclidean_distance_list.append(calc_euclidean_distance(tl_prediction_for_this_pass[day], tl_true_sequence_with_feature_names[day], cut_num_to_lon, cut_num_to_lat))
    euclidean_distance_day_pass.append(euclidean_distance_list)

    tl_hat_for_all_passes = [tl_prediction_for_this_pass]

    transition_matrices = [transition_prob_matrix_forward]

    num_passes = 1
    global pass_num
    pass_num = 1

    # while not end_algorithm:
    for j in range(num_iter-1):

        # run RL-algorithm in other direction
        observed_sequence = observed_sequence[::-1]
        pr_true_sequence_with_feature_names = pr_true_sequence_with_feature_names[::-1]
        tl_true_sequence_with_feature_names = tl_true_sequence_with_feature_names[::-1]

        if direction == 'backward':
            transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, model, tl_prediction_for_this_pass, largest_deviation_for_this_pass = one_pass(
            transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, total_time, delta, lambda_value, 
            observed_sequence, pr_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 tl_state_dict, pr_state_dict, tl_true_sequence_with_feature_names, unique_values_on_each_dimension, cut_num_to_lon, cut_num_to_lat, gamma_attacker)
            transition_matrices.append(transition_prob_matrix_backward)
        else:
            transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, model, tl_prediction_for_this_pass, largest_deviation_for_this_pass = one_pass(
            transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, total_time, delta, lambda_value, 
            observed_sequence, pr_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 tl_state_dict, pr_state_dict, tl_true_sequence_with_feature_names, unique_values_on_each_dimension, cut_num_to_lon, cut_num_to_lat, gamma_attacker)
            transition_matrices.append(transition_prob_matrix_forward)
        
        pass_num += 1
        
        tl_hat_for_all_passes.append(tl_prediction_for_this_pass)
        
        if direction == 'backward':
            euclidean_distance_list = []
            for day in range(len(tl_true_sequence_with_feature_names)):
                euclidean_distance_list.append(calc_euclidean_distance(tl_prediction_for_this_pass[::-1][day], tl_true_sequence_with_feature_names[::-1][day], cut_num_to_lon, cut_num_to_lat))
        
        else:
            euclidean_distance_list = []
            for day in range(len(tl_true_sequence_with_feature_names)):
                euclidean_distance_list.append(calc_euclidean_distance(tl_prediction_for_this_pass[day], tl_true_sequence_with_feature_names[day], cut_num_to_lon, cut_num_to_lat))
                                                                  
        
        euclidean_distance_day_pass.append(euclidean_distance_list)

        if j > 2 * window_size:
            matrices_within_range = transition_matrices[-window_size:]
            stacked_matrices = np.stack(matrices_within_range)
            average_matrix = np.mean(stacked_matrices, axis=0)
            if direction == 'backward':
                transition_prob_matrix_forward = average_matrix
            else:
                transition_prob_matrix_backward = average_matrix
        
        # change direction variable
        if direction == 'forward':
            direction = 'backward'
        else:
            direction = 'forward'
        
        num_passes += 1

    if direction == 'backward':
        return transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, num_passes, model, largest_deviation_for_this_pass, euclidean_distance_day_pass, tl_hat_for_all_passes, tl_prediction_for_this_pass
    else:
        return transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, num_passes, model, largest_deviation_for_this_pass, euclidean_distance_day_pass, tl_hat_for_all_passes, tl_prediction_for_this_pass[::-1]
    
def one_pass(transition_prob_matrix, emission_prob_matrix, initial_distribution, total_time, delta, lambda_value, 
             observed_sequence, pr_true_sequence_with_feature_names, immutable_index,
             num_hidden_states, num_observed_states, tl_state_dict, pr_state_dict, tl_true_sequence_with_feature_names,
             unique_values_on_each_dimension, cut_num_to_lon, cut_num_to_lat, gamma_attacker):
    laplace_factor = 1e-10
    
    tl_prediction_for_this_pass = []
    largest_deviation_for_this_pass = -100
    
    model = hmm.CategoricalHMM(n_components=num_hidden_states, init_params='', random_state=100, n_features=num_observed_states,
                               algorithm='map')
    
    model.transmat_ = normalize(transition_prob_matrix, axis=1, norm = 'l1')
    model.emissionprob_ = normalize(emission_prob_matrix, axis=1, norm = 'l1')
    model.startprob_ = initial_distribution / np.sum(initial_distribution)

    with SuppressPrint():
        model.fit(observed_sequence.reshape(-1,1))

    post_transition_prob_matrix = model.transmat_
    post_emission_prob_matrix = model.emissionprob_
    post_initial_distribution = model.startprob_
    
    # Laplace smoothing to avoid zero-sum rows
    post_transition_prob_matrix += laplace_factor
    post_emission_prob_matrix += laplace_factor

    model.transmat_ = normalize(post_transition_prob_matrix, axis = 1, norm = 'l1')
    model.emissionprob_ = normalize(post_emission_prob_matrix, axis = 1, norm='l1')

    prev_day_rewarded = False
    first_day = True
    tl_prev_hat_index = 0

    pr_true_index = []
    for state in pr_true_sequence_with_feature_names:
        pr_true_index.append(get_pr_index(state, pr_state_dict))
    
    tl_prediction_for_this_pass = []
    
    for i in range(total_time):
        skip_reward = False
        
        # prck the TL state that is a subset of the PR true state with the highest probability
        predicted_sequence_prob = model.predict_proba(observed_sequence.reshape(-1,1))[i]
        tl_i_hat_index = np.argmax(predicted_sequence_prob)
        while (tl_i_hat_index, pr_true_index[i]) in immutable_index:
            predicted_sequence_prob[tl_i_hat_index] = -1
            tl_i_hat_index = np.argmax(predicted_sequence_prob)

        tl_i_hat_state = tl_state_dict[tl_i_hat_index]
        tl_prediction_for_this_pass.append(tl_i_hat_state)

        # predict PR i+1 hat using greedy expansion with TL at center
        pr_i_hat_state_greedy, _ = uniform_attack.expansion_w_tl_at_center_with_deviation(unique_values_on_each_dimension, true_location = tl_i_hat_state, 
                    attack_type= 'PR_uniform_attack', lambda_value = lambda_value, random_seed=100, deviation_amount=0)
        pr_i_hat_state = []
        for dim in pr_i_hat_state_greedy:
            pr_i_hat_state.append(sorted(dim))
        pr_i_hat_index = get_pr_index(pr_i_hat_state, pr_state_dict)

        if pr_i_hat_index == -1:
            skip_reward = True

        deviation = calc_euclidean_distance(tl_i_hat_state, tl_true_sequence_with_feature_names[i], cut_num_to_lon, cut_num_to_lat)
        if deviation > largest_deviation_for_this_pass:
            largest_deviation_for_this_pass=deviation
        
        if not skip_reward:
            post_transition_prob_matrix, post_emission_prob_matrix, prev_day_rewarded = accuracy_reward(pr_i_hat_state, pr_true_sequence_with_feature_names[i], 
                                                                                                                            tl_i_hat_index, tl_prev_hat_index, delta,
                                                            post_transition_prob_matrix, post_emission_prob_matrix, pr_state_dict, prev_day_rewarded, first_day)
            
            
        # prepare for next iteration
        
        model.transmat_ = normalize(post_transition_prob_matrix, axis = 1, norm = 'l1')
       
        model.emissionprob_ = normalize(post_emission_prob_matrix, axis = 1, norm = 'l1')
        model.startprob_ = [0 if np.isnan(prob) else prob for prob in model.startprob_]
        if np.sum(model.startprob_) < 1e-10:
            model.startprob_ = [1/len(model.startprob_) for _ in range(len(model.startprob_))]
        if max(model.startprob_) == 1.0:
            model.startprob_ = [prob + laplace_factor for prob in model.startprob_]
        model.startprob_ = model.startprob_ / np.clip(np.sum(model.startprob_), 1e-10, None)

        # update tl_prev
        tl_prev_hat_index = tl_i_hat_index
        first_day = False

    post_initial_distribution = model.startprob_

    post_transition_prob_matrix = normalize(post_transition_prob_matrix, axis = 1, norm = 'l1')
    post_transition_prob_matrix = post_transition_prob_matrix * (len(post_transition_prob_matrix[0]) * 10)
    post_emission_prob_matrix = normalize(post_emission_prob_matrix, axis = 1, norm = 'l1')
    post_emission_prob_matrix = post_emission_prob_matrix * (len(post_emission_prob_matrix) * 10)
    
    return post_transition_prob_matrix, post_emission_prob_matrix, np.array(post_initial_distribution), model, tl_prediction_for_this_pass, largest_deviation_for_this_pass

def assess_IoU(pr_2_hat_state, pr_2_true_state, delta):
    pr_2_hat_grids = generate_grids_from_pr(pr_2_hat_state)
    pr_2_true_grids = generate_grids_from_pr(pr_2_true_state)

    grid_to_index_dict = {}
    counter = 0
    pr_2_hat_grids_index = []
    for grid in pr_2_hat_grids:
        if grid not in grid_to_index_dict.values():
            grid_to_index_dict[counter] = grid
            pr_2_hat_grids_index.append(counter)
            counter+=1
        else:
            pr_2_hat_grids_index.append(get_pr_index(grid, grid_to_index_dict))
    
    pr_2_true_grids_index = []
    for grid in pr_2_true_grids:
        if grid not in grid_to_index_dict.values():
            grid_to_index_dict[counter] = grid
            pr_2_true_grids_index.append(counter)
            counter+=1
        else:
            pr_2_true_grids_index.append(get_pr_index(grid, grid_to_index_dict))
    pr_2_hat_intersect_pr_2_true = [grid for grid in pr_2_true_grids_index if grid in pr_2_hat_grids_index]
    pr_2_hat_union_pr_2_true = list(set(pr_2_hat_grids_index).union(set(pr_2_true_grids_index)))
    return len(pr_2_hat_intersect_pr_2_true) / len(pr_2_hat_union_pr_2_true) >= delta

def calc_euclidean_distance(tl_2_hat_state, tl_2_true_state, cut_num_to_lon, cut_num_to_lat):
    tl_hat_lon = cut_num_to_lon[int(tl_2_hat_state[0][0])]
    tl_hat_lat = cut_num_to_lat[int(tl_2_hat_state[1][0])]
    tl_true_lon = cut_num_to_lon[int(tl_2_true_state[0][0])]
    tl_true_lat = cut_num_to_lat[int(tl_2_true_state[1][0])]
    
    tl_hat_location = (tl_hat_lat, tl_hat_lon)
    tl_true_location = (tl_true_lat, tl_true_lon)

    return geopy.distance.geodesic(tl_hat_location, tl_true_location).km*1000


def accuracy_reward(pr_2_hat_state, pr_2_true_state, tl_index, tl_prev_hat_index, delta, transition_prob_matrix, emission_prob_matrix, pr_state_dict, prev_day_rewarded, first_day):
    reward_factor = 10000

    # if accuracy constraint satisfied, reward
    if assess_IoU(pr_2_hat_state, pr_2_true_state, delta):
        if prev_day_rewarded and not first_day:
            transition_prob_matrix[tl_prev_hat_index][tl_index] += reward_factor
            
            ## emission reward
            pr_index = get_pr_index(pr_2_true_state, pr_state_dict)
            emission_prob_matrix[tl_index][pr_index] += reward_factor
        
        prev_day_rewarded = True

    # penalize otherwise
    else:
        # transition
        if prev_day_rewarded and not first_day:
            transition_prob_matrix[tl_prev_hat_index][tl_index] = 0
        
            # emission
            pr_index = get_pr_index(pr_2_true_state, pr_state_dict)
            emission_prob_matrix[tl_index][pr_index] = 0

        prev_day_rewarded = False
    
    # normalize and return
    transition_prob_matrix = normalize(transition_prob_matrix, axis = 1, norm = 'l1')
    # Fix near-zero rows in transition matrix
    row_sums = transition_prob_matrix.sum(axis=1)
    bad_rows = (row_sums < 1e-8).ravel()
    if np.any(bad_rows):
        transition_prob_matrix[bad_rows] = 1.0 / transition_prob_matrix.shape[1]

    emission_prob_matrix = normalize(emission_prob_matrix, axis = 1, norm = 'l1')
    # Fix near-zero rows in emission matrix
    row_sums = emission_prob_matrix.sum(axis=1)
    bad_rows = (row_sums < 1e-8).ravel()
    if np.any(bad_rows):
        emission_prob_matrix[bad_rows] = 1.0 / emission_prob_matrix.shape[1]

    return transition_prob_matrix, emission_prob_matrix, prev_day_rewarded
