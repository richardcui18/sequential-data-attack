import numpy as np
import math
import random
from itertools import product
import hmm_rl

def compute_PR_size(published_region):
    PR = list(product(*published_region))
    return len(PR)

def expansion_w_tl_at_center_with_deviation(unique_values_on_each_dimension, true_location, attack_type, lambda_value, random_seed, deviation_amount):
    random.seed(random_seed)
    published_region = true_location.copy()
    deviated = False
    if attack_type == 'PR_uniform_attack':
        # minimum size of PR
        pr_lower_bound = math.ceil(1/lambda_value)
        PR_size = 1
        longitude_upper_limit = max([int(lon) for lon in unique_values_on_each_dimension[0]])
        latitude_upper_limit = max([int(lat) for lat in unique_values_on_each_dimension[1]])
        len_longitude = 1
        len_latitude = 1

        while PR_size < pr_lower_bound:
            longitude_dimension_ints = [int(lon) for lon in published_region[0]]
            latitude_dimension_ints = [int(lat) for lat in published_region[1]]

            # randomly select growing up or left
            direction = random.choice([0, 1])

            possible_growths = []

            # grow in longitude
            if direction == 0:
                if min(longitude_dimension_ints)>0 and max(longitude_dimension_ints)<longitude_upper_limit:
                    possible_growths.append([0, min(longitude_dimension_ints)-1])
                    possible_growths.append([0, max(longitude_dimension_ints)+1])
                    len_longitude += 2

            # grow in latitude
            else:
                if min(latitude_dimension_ints)>0 and max(latitude_dimension_ints)<latitude_upper_limit:
                    possible_growths.append([1, min(latitude_dimension_ints)-1])
                    possible_growths.append([1, max(latitude_dimension_ints)+1])
                    len_latitude += 2
            
            for best_dimension, best_feature in possible_growths:
                # add best_feature to published intent
                published_region_current_dimension = published_region[best_dimension].copy()
                published_region_current_dimension.append(str(best_feature))
                published_region[best_dimension] = published_region_current_dimension
                PR_size = compute_PR_size(published_region)
        
        if len_longitude > len_latitude:
            # negative deviation amount means put TL at the boundary
            if deviation_amount < 0:
                deviation_amount = (len_longitude - 1)//2

            if deviation_amount > 0 and deviation_amount <= (len_longitude - 1)//2:
                deviated = True
                published_region[0] = [str(int(num) - deviation_amount) for num in published_region[0]]

        else:
            # negative deviation amount means put TL at the boundary
            if deviation_amount < 0:
                deviation_amount = (len_latitude - 1)//2

            if deviation_amount > 0 and deviation_amount <= (len_latitude - 1)//2:
                deviated = True
                published_region[1] = [str(int(num) - deviation_amount) for num in published_region[1]]
            
        return published_region, deviated

# helper function to generate all possible subsets of size 1 given a PR state
def generate_subsets_from_pr(state):
    result = []
    for elements in product(*state):
        result.append([[el] for el in elements])
    return result

## Baseline: Uniform attack
def pr_uniform_attack(total_time, pr_true_sequence_with_feature_names, tl_true_sequence_with_feature_names, cut_num_to_lon, cut_num_to_lat):
    random.seed(42)
    predicted_tl_from_em = []
    largest_deviation_baseline = -100

    for i in range(total_time):
        possible_tl_list_time_i = generate_subsets_from_pr(pr_true_sequence_with_feature_names[i])
        tl_i_hat_em_attack = random.choice(possible_tl_list_time_i)
        predicted_tl_from_em.append(tl_i_hat_em_attack)
    
    euclidean_distance_list = []
    for day in range(len(tl_true_sequence_with_feature_names)):
        cur_euclidean_distance = hmm_rl.calc_euclidean_distance(predicted_tl_from_em[day], tl_true_sequence_with_feature_names[day], cut_num_to_lon, cut_num_to_lat)
        euclidean_distance_list.append(cur_euclidean_distance)
        if cur_euclidean_distance > largest_deviation_baseline:
            largest_deviation_baseline = cur_euclidean_distance
    mean_euclidean_distance = np.mean(np.array(euclidean_distance_list))
    
    return euclidean_distance_list, mean_euclidean_distance, predicted_tl_from_em, largest_deviation_baseline