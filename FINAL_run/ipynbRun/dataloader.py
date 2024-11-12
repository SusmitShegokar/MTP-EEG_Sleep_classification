import os
import mne
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_FOLDER ='/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/'
SOURCE_DATA_FOLDER = DATA_FOLDER  + "SleepSource/"
LEADERBOARD_TARGET_DATA_FOLDER = DATA_FOLDER + "LeaderboardSleep/sleep_target/"
LEADERBOARD_TEST_DATA_FOLDER = DATA_FOLDER + "LeaderboardSleep/testing/"
FINAL_TARGET_DATA_FOLDER = DATA_FOLDER + "finalSleep/sleep_target/"
FINAL_TEST_DATA_FOLDER = DATA_FOLDER + "finalSleep/testing/"



def load_data(data_folder, load_labels=True):
    fn_list = sorted(os.listdir(data_folder))
    print(f"Loading data from folder: {data_folder} ({len(fn_list)} files) - Load labels {load_labels}")

    data_map = {}
    subject_list = []
    sample_counter = 0

    for fn in tqdm(fn_list):
        if fn.endswith("X.npy"):
            code = fn.split("_")[1][:-4]
        elif fn == "headerInfo.npy":
            meta = np.load(data_folder + fn, allow_pickle=True)
            print(meta)
            continue
        else:
            continue

        eeg = np.load(data_folder + fn, allow_pickle=True)

        if load_labels:
            label_fn = fn.replace("X", "y")
            label = np.load(data_folder + label_fn, allow_pickle=True)
        else:
            label = None

        s_part, r_part = code.split("r")
        subject = int(s_part[1:])
        repetition = int(r_part[:-1])

        subject_list.append(subject)

        if subject not in data_map.keys():
            data_map[subject] = {}

        data_map[subject][repetition] = {"eeg": eeg, "label": label}
        sample_counter += len(eeg)

    subject_list = np.unique(subject_list)
    print(f"Loaded total {sample_counter} samples for subjects: {subject_list}")
    return data_map, subject_list

def prepare_window_data(data, subject_list=None):
    window_data = []
    window_labels = []

    if subject_list is None:
        subject_list = data.keys()

    for s in tqdm(subject_list):
        for r in data[s].keys():
            eeg = data[s][r]["eeg"]
            label = data[s][r]["label"]

            window_data.extend(eeg)

            if label is not None:
                window_labels.extend(label)

    return window_data, window_labels

def print_stats(desc, data):
    print(f"{desc} mean: {np.mean(data)}, std: {np.std(data)}, min: {np.min(data)}, max: {np.max(data)}")

def normalize(data, mean_value, std_value, desc=""):
    data = np.array(data)
    data = (data - mean_value) / std_value
    print_stats(desc, data)
    return list(data)

# def filter_freq(data, f_min, f_max, FS):
#     return mne.filter.filter_data(np.array(data, dtype=np.float64), FS, f_min, f_max, method="iir", verbose=False)

# def downsample(data, FS, FS_new):
#     return mne.filter.resample(data, down=FS/FS_new)



# def seed_everything(seed):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

def get_phase_1_data():
    source_data, source_subjects = load_data(SOURCE_DATA_FOLDER)
    source_data, source_labels = prepare_window_data(source_data, source_subjects)
    # calculate stats of source data and normalize it
    source_data = np.array(source_data)
    source_mean = np.mean(source_data)
    source_std = np.std(source_data)
    source_data = (source_data - source_mean) / source_std
    print(f"Source mean: {source_mean}, std: {source_std}, min: {np.min(source_data)}, max: {np.max(source_data)}")
    source_data  = list(source_data)
    # load and normalize target data
    lb_target_data, lb_target_subjects = load_data(LEADERBOARD_TARGET_DATA_FOLDER)
    lb_target_data, lb_target_labels = prepare_window_data(lb_target_data, lb_target_subjects)
    lb_target_data = normalize(lb_target_data, source_mean, source_std, "Leadeboard target")
    # load and normalize test data
    lb_test_data, lb_test_subjects = load_data(LEADERBOARD_TEST_DATA_FOLDER, load_labels=False)
    lb_test_data, lb_test_labels = prepare_window_data(lb_test_data, lb_test_subjects)
    lb_test_data = normalize(lb_test_data, source_mean, source_std, "Leadeboard test")
    
        # load and normalize target data
    fn_target_data, fn_target_subjects = load_data(FINAL_TARGET_DATA_FOLDER)
    fn_target_data, fn_target_labels = prepare_window_data(fn_target_data, fn_target_subjects)
    fn_target_data = normalize(fn_target_data, source_mean, source_std, "Final target")
    # load and normalize test data
    fn_test_data, fn_test_subjects = load_data(FINAL_TEST_DATA_FOLDER, load_labels=False)
    fn_test_data, fn_test_labels = prepare_window_data(fn_test_data, fn_test_subjects)
    fn_test_data = normalize(fn_test_data, source_mean, source_std, "Final test")

    return source_data, source_labels, lb_target_data, lb_target_labels, lb_test_data, lb_test_data,fn_target_data, fn_target_labels ,fn_test_data, fn_test_labels



def split_train_val_test(train_data, train_labels, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split

    # First, split into training and temp (validation + test)
    train_data_split, temp_data, train_labels_split, temp_labels = train_test_split(
        train_data, train_labels, train_size=train_size, random_state=random_state, stratify=train_labels
    )

    # Calculate the proportion of validation and test sizes relative to temp_data
    temp_val_size = val_size / (val_size + test_size)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=1 - temp_val_size, random_state=random_state, stratify=temp_labels
    )

    return train_data_split, val_data, test_data, train_labels_split, val_labels, test_labels

def create_data_directories(base_dir='data', subdirs=['train', 'validate', 'test']):
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")

def save_splits(base_dir, splits):
    """
    Saves the data and labels into respective directories.

    Parameters:
    - base_dir: The main directory containing subdirectories.
    - splits: A tuple containing (train_data, val_data, test_data, train_labels, val_labels, test_labels).
    """
    train_data, val_data, test_data, train_labels, val_labels, test_labels = splits
    splits_dict = {
        'train': (train_data, train_labels),
        'validate': (val_data, val_labels),
        'test': (test_data, test_labels)
    }

    for split_name, (data, labels) in splits_dict.items():
        data_path = os.path.join(base_dir, split_name, 'data.npy')
        labels_path = os.path.join(base_dir, split_name, 'labels.npy')
        np.save(data_path, data)
        np.save(labels_path, labels)
        print(f"Saved {split_name} data to {data_path} and labels to {labels_path}")

def save_remaining_variables():
    # Create necessary directories
    directories = ['data/leader', 'data/final']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save leaderboard data
    np.save('data/leader/lbtarget_data.npy', lb_target_data)
    np.save('data/leader/lbtarget_labels.npy', lb_target_labels)
    np.save('data/leader/test_data.npy', lb_test_data)
    np.save('data/leader/test_data_cpy.npy', lb_test_data_cpy)

    # Save final data
    np.save('data/final/fntarget_data.npy', fn_target_data)
    np.save('data/final/fntarget_labels.npy', fn_target_labels)
    np.save('data/final/test_data.npy', fn_test_data)
    np.save('data/final/test_data_cpy.npy', fn_test_data_cpy)

    print("All variables saved successfully.")


source_data, source_labels, lb_target_data, lb_target_labels, lb_test_data, lb_test_data_cpy, fn_target_data, fn_target_labels, fn_test_data, fn_test_data_cpy= get_phase_1_data()

tmp = np.array(source_data)
supervised_mixup_data = {}
for c in np.unique(source_labels):
    supervised_mixup_data[c] = tmp[source_labels == c]
    print(c, np.shape(supervised_mixup_data[c]))

del tmp

splits = split_train_val_test(source_data, source_labels)
create_data_directories()
save_splits('data', splits)
save_remaining_variables()
