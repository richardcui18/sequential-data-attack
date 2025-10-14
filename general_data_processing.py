import pandas as pd
import numpy as np
import os
import itertools
import csv
import ast
import hmm_rl

# ======================
# Dataset Readers
# ======================
def read_geolife_trajectory(person_id, trajectory_id, sample_frequency_minutes, max_frequency_minutes=1):
    file_path = f"data/geolife/{person_id}/Trajectory/{trajectory_id}.plt"
    df = pd.read_csv(file_path, skiprows=6, delimiter=',')
    df.columns = ['Latitude', 'Longitude', '0', 'Altitude', 'Time past 12/40/1899', 'Date', 'Time']
    df.drop(columns=['0', 'Altitude'], inplace=True)

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    return _resample_trajectory(df, sample_frequency_minutes, max_frequency_minutes)


def read_taxi_porto_trajectory(trip_id, sample_frequency_minutes, max_frequency_minutes=1):
    file_path = "data/taxi_porto/train.csv"
    df = pd.read_csv(file_path)

    trip = df[df['TRIP_ID'] == trip_id].iloc[0]
    polyline = ast.literal_eval(trip['POLYLINE'])
    if len(polyline) == 0:
        return pd.DataFrame(columns=["Latitude", "Longitude", "Datetime"])

    lons, lats = zip(*polyline)
    datetimes = pd.date_range(start=pd.to_datetime(trip['TIMESTAMP'], unit='s'),
                              periods=len(polyline), freq='15s')

    df_out = pd.DataFrame({"Latitude": lats, "Longitude": lons, "Datetime": datetimes})
    return _resample_trajectory(df_out, sample_frequency_minutes, max_frequency_minutes)

import pickle

def read_synmob_trajectory(dataset, traj_idx, sample_frequency_minutes, max_frequency_minutes=1):
    if dataset == "synmob_xian":
        file_path = "data/synmob/trajs_Xian-500.pkl"
    elif dataset == "synmob_chengdu":
        file_path = "data/synmob/trajs_Chengdu-500.pkl"
    else:
        raise ValueError("Unsupported synmob dataset")

    with open(file_path, "rb") as f:
        trajs = pickle.load(f)

    if traj_idx >= len(trajs):
        return pd.DataFrame(columns=["Latitude", "Longitude", "Datetime"])

    traj = trajs[traj_idx]
    lons, lats = zip(*traj)

    datetimes = pd.date_range("2020-01-01", periods=len(traj), freq=f"{int(sample_frequency_minutes*60)}s")

    df = pd.DataFrame({"Latitude": lats, "Longitude": lons, "Datetime": datetimes})
    return _resample_trajectory(df, sample_frequency_minutes, max_frequency_minutes)


# ======================
# Shared Resampler
# ======================
def _resample_trajectory(df, sample_frequency_minutes, max_frequency_minutes):
    if df.empty:
        return df

    filtered_rows = []
    last_time = df['Datetime'].iloc[0]
    filtered_rows.append(df.iloc[0])

    for index in range(1, len(df)):
        current_time = df['Datetime'].iloc[index]
        if pd.Timedelta(minutes=max_frequency_minutes) >= (current_time - last_time) \
           and (current_time - last_time) >= pd.Timedelta(minutes=sample_frequency_minutes):
            filtered_rows.append(df.iloc[index])
            last_time = current_time

    return pd.DataFrame(filtered_rows)


# ======================
# Dispatcher
# ======================
def read_one_trajectory(dataset, *ids, sample_frequency_minutes=1, max_frequency_minutes=1):
    if dataset == "geolife":
        return read_geolife_trajectory(*ids, sample_frequency_minutes, max_frequency_minutes)
    elif dataset == "taxi_porto":
        return read_taxi_porto_trajectory(*ids, sample_frequency_minutes, max_frequency_minutes)
    elif dataset in ["synmob_xian", "synmob_chengdu"]:
        return read_synmob_trajectory(dataset, *ids, sample_frequency_minutes, max_frequency_minutes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ======================
# Trajectory Listings
# ======================
def list_all_trajectory_files_geolife(person_id):
    folder_path = f"data/geolife/{person_id}/Trajectory/"
    return [f for f in os.listdir(folder_path) if f.endswith('.plt')]

def list_all_trips_taxi_porto():
    file_path = "data/taxi_porto/train.csv"
    df = pd.read_csv(file_path, usecols=['TRIP_ID'])
    return df['TRIP_ID'].tolist()

def list_all_trajs_synmob(dataset):
    if dataset == "synmob_xian":
        file_path = "data/synmob/trajs_Xian-500.pkl"
    elif dataset == "synmob_chengdu":
        file_path = "data/synmob/trajs_Chengdu-500.pkl"
    else:
        raise ValueError("Unsupported synmob dataset")

    with open(file_path, "rb") as f:
        trajs = pickle.load(f)
    return list(range(len(trajs)))


# ======================
# Save / Load
# ======================
def save_to_csv(data, file_name, column_names, folder_name='data/processed'):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, file_name)
    print(f"Process trajectory data saved to {folder_name}/{file_name}.")
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(data)


# ======================
# Processed Data Sampler
# ======================
def get_processed_data(dataset, num_trajectories):
    if dataset == "geolife":
        processed_data = pd.read_csv('data/geolife/processed/processed_geographic_range')
        processed_data['Person ID'] = processed_data.index.map(lambda x: f"{x:03d}")
        processed_data['Trajectory IDs'] = processed_data['Trajectory IDs'].map(str)
        processed_data['ID Pairs'] = list(zip(processed_data['Person ID'], processed_data['Trajectory IDs']))
        return processed_data.sample(n=num_trajectories, random_state=1)

    elif dataset == "taxi_porto":
        df = pd.read_csv("data/taxi_porto/train.csv", usecols=['TRIP_ID'])
        sampled = df.sample(n=num_trajectories, random_state=1)
        sampled['ID Pairs'] = list(zip(sampled['TRIP_ID']))
        return sampled
    elif dataset in ["synmob_xian", "synmob_chengdu"]:
        ids = list_all_trajs_synmob(dataset)
        sampled = np.random.choice(ids, num_trajectories, replace=False)
        df = pd.DataFrame({'ID Pairs': [(tid,) for tid in sampled]})
        return df
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ======================
# Bounding Boxes
# ======================
def calculate_one_bounding_box(trajectory_df):
    longitudes = trajectory_df['Longitude'].tolist()
    latitudes = trajectory_df['Latitude'].tolist()
    bounding_box_lon = [min(longitudes)-0.01, max(longitudes)+0.01]
    bounding_box_lat = [min(latitudes)-0.01, max(latitudes)+0.01]
    return [bounding_box_lon, bounding_box_lat]

def calculate_max_bounding_box(dataset, trajectory_ids_pairs, sample_frequency_minutes):
    max_bounding_box_lon_range = [float('inf'), float('-inf')]
    max_bounding_box_lat_range = [float('inf'), float('-inf')]
    for ids in trajectory_ids_pairs:
        trajectory_df = read_one_trajectory(dataset, *ids, sample_frequency_minutes=sample_frequency_minutes)
        if trajectory_df.empty:
            continue
        bounding_box_lon_range, bounding_box_lat_range = calculate_one_bounding_box(trajectory_df)
        max_bounding_box_lon_range[0] = min(max_bounding_box_lon_range[0], bounding_box_lon_range[0])
        max_bounding_box_lon_range[1] = max(max_bounding_box_lon_range[1], bounding_box_lon_range[1])
        max_bounding_box_lat_range[0] = min(max_bounding_box_lat_range[0], bounding_box_lat_range[0])
        max_bounding_box_lat_range[1] = max(max_bounding_box_lat_range[1], bounding_box_lat_range[1])
    return max_bounding_box_lon_range, max_bounding_box_lat_range


# ======================
# Grid Calculations
# ======================
def lat_lon_to_area(lat_range, lon_range, reference_latitude=39.990):
    height = lat_range * 111139
    lon_m_per_degree = 111319 * np.cos(np.radians(reference_latitude))
    width = lon_range * lon_m_per_degree
    return height * width

def calculate_cut_lengths(dataset, id_pairs, sample_frequency_minutes,
                          trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit):
    longtiude_cut_length = trajectory_grid_resolution_lon_limit
    latitude_cut_length = trajectory_grid_resolution_lat_limit
    longtiude_cut_length = 0.1
    latitude_cut_length = 0.1
    max_in_same_grid_lat = 1
    max_in_same_grid_lon = 1

    for ids in id_pairs:
        trajectory_df = read_one_trajectory(dataset, *ids, sample_frequency_minutes=sample_frequency_minutes)
        if trajectory_df.empty:
            continue

        longitudes = trajectory_df['Longitude'].tolist()
        pairwise_diffs_lon = [abs(a - b) for a, b in itertools.combinations(longitudes, 2)]
        if pairwise_diffs_lon:
            pairwise_diffs_sorted_lon = sorted(pairwise_diffs_lon)
            min_diff_lon = pairwise_diffs_sorted_lon[0]
            if min_diff_lon < longtiude_cut_length:
                i = 0
                while i < len(pairwise_diffs_sorted_lon) and pairwise_diffs_sorted_lon[i] <= trajectory_grid_resolution_lon_limit:
                    i += 1
                if i < len(pairwise_diffs_sorted_lon):
                    longtiude_cut_length = pairwise_diffs_sorted_lon[i]
                if i > max_in_same_grid_lon:
                    max_in_same_grid_lon = i

        latitudes = trajectory_df['Latitude'].tolist()
        pairwise_diffs_lat = [abs(a - b) for a, b in itertools.combinations(latitudes, 2)]
        if pairwise_diffs_lat:
            pairwise_diffs_sorted_lat = sorted(pairwise_diffs_lat)
            min_diff_lat = pairwise_diffs_sorted_lat[0]
            if min_diff_lat < latitude_cut_length:
                i = 0
                while i < len(pairwise_diffs_sorted_lat) and pairwise_diffs_sorted_lat[i] <= trajectory_grid_resolution_lat_limit:
                    i += 1
                if i < len(pairwise_diffs_sorted_lat):
                    latitude_cut_length = pairwise_diffs_sorted_lat[i]
                if i > max_in_same_grid_lat:
                    max_in_same_grid_lat = i

    print('Area of one grid (m^2):', lat_lon_to_area(latitude_cut_length, longtiude_cut_length))
    return longtiude_cut_length, latitude_cut_length


# ======================
# TL Sequence Creator
# ======================
def create_tl_sequence(dataset, sample_frequency_minutes,
                       trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit,
                       num_trajectories):
    processed_df = get_processed_data(dataset, num_trajectories)
    id_pairs = list(processed_df['ID Pairs'])
    bounding_box = calculate_max_bounding_box(dataset, id_pairs, sample_frequency_minutes)
    bounding_box_lon, bounding_box_lat = bounding_box

    theoretical_max_without_constraint = 0
    longitude_cut_length, latitude_cut_length = calculate_cut_lengths(
        dataset, id_pairs, sample_frequency_minutes,
        trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit
    )

    lon_start, lon_end = bounding_box_lon
    lat_start, lat_end = bounding_box_lat
    num_lon_cuts = int(np.ceil((lon_end - lon_start) / longitude_cut_length))
    num_lat_cuts = int(np.ceil((lat_end - lat_start) / latitude_cut_length))

    cut_num_to_lon = np.linspace(lon_start, lon_end, num_lon_cuts + 1)
    cut_num_to_lat = np.linspace(lat_start, lat_end, num_lat_cuts + 1)

    unique_values_on_each_dimension = [
        [str(i) for i in range(num_lon_cuts)],
        [str(i) for i in range(num_lat_cuts)]
    ]

    data_cube = np.ones((num_lon_cuts, num_lat_cuts))
    cost_cube = data_cube.copy()

    tl_true_sequences = []
    tl_true_sequences_with_feature_names = []

    for ids in id_pairs:
        trajectory_df = read_one_trajectory(dataset, *ids, sample_frequency_minutes=sample_frequency_minutes)
        if trajectory_df.empty:
            continue

        lon_sequence = trajectory_df['Longitude'].tolist()
        lat_sequence = trajectory_df['Latitude'].tolist()

        tl_sequence = []
        tl_sequence_with_names = []

        for lon, lat in zip(lon_sequence, lat_sequence):
            lon_index = find_grid_index(lon, lon_start, longitude_cut_length, num_lon_cuts)
            lat_index = find_grid_index(lat, lat_start, latitude_cut_length, num_lat_cuts)

            tl_sequence.append([[lon_index, lat_index]])
            tl_sequence_with_names.append([[str(lon_index)], [str(lat_index)]])

            cur_theoretical_max = max(
                hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [['0'], ['0']], cut_num_to_lon, cut_num_to_lat),
                hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [[str(len(cut_num_to_lon)-1)], ['0']], cut_num_to_lon, cut_num_to_lat),
                hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [['0'], [str(len(cut_num_to_lat)-1)]], cut_num_to_lon, cut_num_to_lat),
                hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]],
                                               [[str(len(cut_num_to_lon)-1)], [str(len(cut_num_to_lat)-1)]],
                                               cut_num_to_lon, cut_num_to_lat)
            )
            theoretical_max_without_constraint = max(theoretical_max_without_constraint, cur_theoretical_max)
        
        if dataset == "geolife":
            MAX_SEQ_LEN = 20
        elif dataset == "taxi_porto":
            MAX_SEQ_LEN = 10
        elif dataset in ['synmob_xian', 'synmob_chengdu']:
            MAX_SEQ_LEN = 10

        if len(tl_sequence) > MAX_SEQ_LEN:
            tl_sequence = tl_sequence[:MAX_SEQ_LEN]
            tl_sequence_with_names = tl_sequence_with_names[:MAX_SEQ_LEN]

        tl_true_sequences.append(tl_sequence)
        tl_true_sequences_with_feature_names.append(tl_sequence_with_names)

    return (unique_values_on_each_dimension, data_cube, cost_cube,
            tl_true_sequences, tl_true_sequences_with_feature_names,
            cut_num_to_lon, cut_num_to_lat, theoretical_max_without_constraint)


# ======================
# Helper
# ======================
def find_grid_index(value, start, cut_length, num_cuts):
    index = int((value - start) / cut_length)
    return min(max(0, index), num_cuts - 1)
