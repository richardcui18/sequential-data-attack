import random
import numpy as np
import general_data_processing
import uniform_attack
from train_seqgan_trajectories import generate_synthetic_trajectories

def get_dataset(dataset_name, deviation_amount_user, lambda_value):
    # Setup
    sample_frequency_minutes = 0.3
    trajectory_grid_resolution_lon_limit = 0.001
    trajectory_grid_resolution_lat_limit = 0.001
    seed = 65

    if dataset_name == 'geolife':
        num_trajectories = 658
    elif dataset_name == 'taxi_porto':
        num_trajectories = 500
    elif dataset_name == 'synmob_xian':
        num_trajectories = 500
    elif dataset_name == 'synmob_chengdu':
        num_trajectories = 500
    
    dataset = {
        'unique_values_on_each_dimension': [],
        'tl_true_sequences': [],
        'tl_true_sequences_with_feature_names': [],
        'pr_true_sequences_with_feature_names': [],
        'total_times': [],
        'cut_num_to_lon': [],
        'cut_num_to_lat': []
    }

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create true TL sequence
    if dataset_name in ['geolife', 'taxi_porto', 'synmob_xian', 'synmob_chengdu']:
        (unique_values_on_each_dimension, _, _, 
         tl_true_sequences, tl_true_sequences_with_feature_names,
         cut_num_to_lon, cut_num_to_lat, _) = general_data_processing.create_tl_sequence(
            dataset_name,
            sample_frequency_minutes,
            trajectory_grid_resolution_lon_limit,
            trajectory_grid_resolution_lat_limit,
            num_trajectories
        )

    # Create true PR sequence
    pr_true_sequences_with_feature_names = []
    total_times = []

    deviation_prop_list = []

    for i in range(len(tl_true_sequences)):
        tl_true_sequence_with_feature_names = tl_true_sequences_with_feature_names[i]

        pr_true_sequence_with_feature_names = []

        deviated_num = 0

        for i, tl_i in enumerate(tl_true_sequence_with_feature_names):
            pr_i, deviated = uniform_attack.expansion_w_tl_at_center_with_deviation(unique_values_on_each_dimension, true_location = tl_i, 
                        attack_type= 'PR_uniform_attack', lambda_value = lambda_value, random_seed=seed+i, deviation_amount=deviation_amount_user)
            pr_true_sequence_with_feature_names.append([sorted(pr_i[0], key=int), sorted(pr_i[1], key=int)])

            if deviated:
                deviated_num += 1
        
        deviation_prop = deviated_num / len(tl_true_sequence_with_feature_names)
        deviation_prop_list.append(deviation_prop)
        
        pr_true_sequences_with_feature_names.append(pr_true_sequence_with_feature_names)
    
        total_times.append(len(tl_true_sequence_with_feature_names))

    print("Mean deviation proprtion:", np.mean(np.array(deviation_prop_list)))
    
    dataset['unique_values_on_each_dimension'] = unique_values_on_each_dimension
    dataset['pr_true_sequences_with_feature_names'] = pr_true_sequences_with_feature_names
    dataset['total_times'] = total_times
    dataset['tl_true_sequences'] = tl_true_sequences
    dataset['tl_true_sequences_with_feature_names'] = tl_true_sequences_with_feature_names
    dataset['cut_num_to_lon'] = cut_num_to_lon
    dataset['cut_num_to_lat'] = cut_num_to_lat

    return dataset