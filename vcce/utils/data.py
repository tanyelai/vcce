import os
import ast
import json
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class MyDictionary(dict):
    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


# Functions for create_test_patient
def load_data(path):
    signals_train = np.load(os.path.join(path, 'signals_train.npy'))
    labels_train = np.load(os.path.join(path, 'labels_train.npy'))
    patient_info_train = np.load(os.path.join(path, 'patient_info_train.npy'))
    return signals_train, labels_train, patient_info_train


def retrieve_test_patient(path, patient_idx, output_dir):
    
    signals, labels, infos = load_data(path)

    idx = np.where(infos[:, 2] == patient_idx)[0]

    label = labels[idx[0]]
    info = infos[idx[0]]
    signal = signals[idx[0]]
    
    print(f"Note: Selected index is {label} patient, and the patient info is [Age: {info[0]}, Gender: {info[1]}, Patient ID: {info[2]}]: ")

    # Derive the output_filename based on patient_idx and label
    output_filename = f"{patient_idx}_{label}"

    # Create the dataset directory
    dataset_dir = os.path.join(output_dir, output_filename)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"The extracted patients are saved in the following directory:\n\
          {dataset_dir}\n")
 
    # Save train and test data to separate files
    np.save(f"{dataset_dir}/signal.npy", signal)
    np.save(f"{dataset_dir}/label.npy", label)
    np.save(f"{dataset_dir}/info.npy", info)


# Functions for create_dataset
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
    else:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic, agg_df):
    new_dict = MyDictionary()
    for key, value in y_dic.items():
        if key in agg_df.index and key not in ['INJAS', 'INJAL', 'INJIN', 'INJLA', 'INJIL', 'PMI']:
            new_key = agg_df.loc[key].diagnostic_class
            new_dict.add(new_key, int(value))
    return [new_dict]


def retrieve_exact_diagnoses(listx):
    return list(set(key for key, value in listx[0].items() if value >= 99.9))



def get_new_data(X, Y):
    x_new, y_new, patient_info = [], [], []
    for x, y, age_sex_id in zip(X, Y.diagnostic_superclass.values, Y[['age', 'sex', 'patient_id']].values):
        if len(y) > 0:
            if 'NORM' in y and len(y) == 1:
                x_new.append(x)
                patient_info.append(age_sex_id)
                y_new.append("NORM")
            elif 'MI' in y and "STTC" not in y:
                x_new.append(x)
                patient_info.append(age_sex_id)
                y_new.append("MI")
    return x_new, y_new, patient_info


def create_binary_dataset(x_data, y_data, class_labels, patient_info, random_state):
    # Find the class with the minimum count
    min_label_count = min(Counter(y_data).values())

    # Undersample the data for each class
    equalized_data = []
    equalized_info = []
    for class_label in class_labels:
        class_indices = np.where(y_data == class_label)[0]
        resampled_indices = resample(class_indices, n_samples=min_label_count, replace=False, random_state=random_state)

        resampled_data = x_data[resampled_indices]
        resampled_info = patient_info[resampled_indices]

        equalized_data.append(resampled_data)
        equalized_info.append(resampled_info)

    x_equalized = np.vstack(equalized_data)
    y_equalized = np.hstack([np.full(min_label_count, class_label) for class_label in class_labels])
    info_equalized = np.vstack(equalized_info)

    print("\n\n")
    print(f"Class labels: {class_labels}")
    class_counts = {label: int(count) for label, count in Counter(y_equalized).items()}
    print(f"Class counts: {class_counts}")
    return x_equalized, y_equalized, info_equalized 


def create(x_new, y_new, patient_info, dataset_dir):
    x_equalized, y_equalized, info_equalized = create_binary_dataset(x_new, y_new, ["NORM", "MI"], patient_info, random_state=0)
    
    # Split equalized data into train and test sets
    x_train, x_test, y_train, y_test, info_train, info_test = train_test_split(
        x_equalized, y_equalized, info_equalized, test_size=0.2, stratify=y_equalized, random_state=0)

    # Create the dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Save train and test data to separate files
    np.save(f"{dataset_dir}/signals_train.npy", x_train)
    np.save(f"{dataset_dir}/labels_train.npy", y_train)
    np.save(f"{dataset_dir}/patient_info_train.npy", info_train)
    np.save(f"{dataset_dir}/signals_test.npy", x_test)
    np.save(f"{dataset_dir}/labels_test.npy", y_test)
    np.save(f"{dataset_dir}/patient_info_test.npy", info_test)

    # Count the occurrences of the class label in the train and test sets
    class_counts_train = {label: int(count) for label, count in Counter(y_train).items()}
    class_counts_test = {label: int(count) for label, count in Counter(y_test).items()}
    
    # Save train and test class counts as text (txt) files
    with open(f"{dataset_dir}/class_counts_train.txt", "w") as f_train:
        f_train.write(json.dumps(class_counts_train, indent=4))
        
    with open(f"{dataset_dir}/class_counts_test.txt", "w") as f_test:
        f_test.write(json.dumps(class_counts_test, indent=4))


def create_dataset(path, sampling_rate, output_dir):
    
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic, agg_df=agg_df)
    Y['diagnostic_superclass'] = Y.diagnostic_superclass.apply(retrieve_exact_diagnoses)

    x_new, y_new, patient_info = get_new_data(X, Y)

    create(np.array(x_new), np.array(y_new), np.array(patient_info), output_dir)
