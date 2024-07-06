import os
import time
from tkinter import filedialog
from icecream import ic
import numpy as np
import pandas as pd
from pandas import DataFrame as DF #often used so shortened alias
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

sequence_length = 10

dataset_interpretation: dict = {
    'gender': {
        1: 'female',
        2: 'male'
    },
    'afftype': {
        1: 'bipolar II',
        2: 'unipolar depressive',
        3: 'bipolar I'
    },
    'melanch': {
        1: 'melancholia',
        2: 'no melancholia'
    },
    'inpatient': {
        1: 'inpatient',
        2: 'outpatient'
    },
    'marriage': {
        1: 'married or cohabiting',
        2: 'single'
    },
    'work': {
        1: 'working or studying',
        2: 'unemployed/sick leave/pension'
    }
}

dataset_interpretation_reversed: dict = {
    'age': {
        '20-24': 1,
        '25-29': 2,
        '30-34': 3,
        '35-39': 4,
        '40-44': 5,
        '45-49': 6,
        '50-54': 7,
        '55-59': 8,
        '60-64': 9,
        '65-69': 10,
    },
    'gender': {
        'female':1,
        'male':2,
    },
    'work': {
        'working or studying':1,
        'unemployed/sick leave/pension':2
    }
}

age_mapping = {
    '20-24': 1,
    '25-29': 2,
    '30-34': 3,
    '35-39': 4,
    '40-44': 5,
    '45-49': 6,
    '50-54': 7,
    '55-59': 8,
    '60-64': 9,
    '65-69': 10,
}


top_path = rf'{os.getcwd()}' # raw strings allow consistent slashes
ic(top_path)

#datapath = filedialog.askopenfile("Locate top level folder containing activity data")
data_path = top_path + r'/data'

def interpret_values(row, conversions):
    for category in conversions:
        if category in row:
            if category == 'age':
                row[category] = str(conversions[category].get(row[category], row[category]))
            else:
                row[category] = conversions[category].get(row[category], row[category])
    return row

condition_numbers: list = []

def load_condition_data(scores_data_interpreted):
    numbers = scores_data_interpreted['number']
    global condition_numbers # call and edit the global variable
    condition_numbers = [item for item in numbers if not item.startswith('control')]
    condition: dict = {}
    condition_path = data_path + r'/condition'

    for num in condition_numbers:
        condition_path_num = condition_path + rf'/{num}.csv'
        activity_data_temp = pd.read_csv(condition_path_num)
        new_activity_data_temp = DF()

        new_activity_data_temp['timestamp'] = pd.to_datetime(activity_data_temp['timestamp'])
        new_activity_data_temp['time_since_start[mins]'] = (new_activity_data_temp['timestamp'] - \
            new_activity_data_temp['timestamp'].iloc[0]).dt.total_seconds() / 60.0
        new_activity_data_temp['activity'] = activity_data_temp['activity']

        #ic(activity_data_temp.head())
        condition[num] = new_activity_data_temp

    #ic(condition)
    return condition



def load_scores() -> DF:
    scores_path = data_path + r'/scores.csv'
    scores_data = pd.read_csv(scores_path)
    ic(scores_data.head()) # check first few records in terminal output

    scores_data_interpreted = DF()
    for i,row in scores_data.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation)
        #ic(row_interpreted)
        scores_data_interpreted = pd.concat([scores_data_interpreted, row_interpreted], axis=1)

    scores_data_interpreted = scores_data_interpreted.T #transpose

    #ic(scores_data_interpreted.head())
    return scores_data_interpreted

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

def scale_and_prepare(scores:DF=None,condition:dict=None):
    # Scale the activity data
    global sequence_length
    scalers = {}
    patient_scaled_data = {}
    X_time_series = {}

    for patient_id, patient_df in condition.items():
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(patient_df['activity']).reshape(-1, 1))
        scalers[patient_id] = scaler
        patient_scaled_data[patient_id] = scaled_data

    for patient_id, scaled_data in patient_scaled_data.items():
        X_time_series[patient_id] = create_sequences(scaled_data, sequence_length)

    key_predictors = ['number', 'age', 'gender', 'madrs1', 'madrs2']
    #ic(scores.head())

    demographic_temp: DF = scores[scores['number'].str.startswith('condition')]
    #ic(demographic_temp.head())

    demographic = DF()
    for i,field in enumerate(key_predictors):
        demographic = pd.concat([demographic, demographic_temp[field]], axis=1)

        #filter out all rows and columns except     number condition_n, age, gender, madrs1, madrs2
    ic(demographic.head())

    demographic_encoded = DF() # re-encode data back to integers
    for i, row in demographic.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation_reversed)
        #print(f"{row=}, {row_interpreted=}")
        demographic_encoded = pd.concat([demographic_encoded, row_interpreted], axis=1)
    demographic_encoded = demographic_encoded.T

    ic(demographic_encoded.head())
    time.sleep(5)

    return patient_scaled_data, demographic_encoded, X_time_series

def train_LSTM(scores: DF = None, condition: dict = None):
    patient_scaled_activity_dict, demographic_refined, X_time_series = scale_and_prepare(scores=scores, condition=condition)
    # Prepare final input for LSTM for each patient
    X_combined = {}

    for patient_id, sequences in X_time_series.items():
        demographic_data = X_demographic.loc[X_demographic.index == patient_id].values
        demographic_data = np.repeat(demographic_data[:, np.newaxis, :], sequences.shape[1], axis=1)
        X_combined[patient_id] = np.concatenate((sequences, demographic_data), axis=2)


if __name__ == '__main__':
    scores_df = load_scores() #dataframe of scores
    condition_dict_df = load_condition_data(scores_df) #dict of key=condition_n, value=dataframe activity time series
        #cols = timestamp, time_since_start[mins], activity
    train_LSTM(scores=scores_df, condition=condition_dict_df)
