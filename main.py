import os
import time
from tkinter import filedialog
from icecream import ic
import numpy as np
import pandas as pd
from pandas import DataFrame as DF  # often used so shortened alias
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error

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
        '20-24': 1.0,
        '25-29': 2.0,
        '30-34': 3.0,
        '35-39': 4.0,
        '40-44': 5.0,
        '45-49': 6.0,
        '50-54': 7.0,
        '55-59': 8.0,
        '60-64': 9.0,
        '65-69': 10.0,
    },
    'gender': {
        'female': 1.0,
        'male': 2.0,
    },
    'work': {
        'working or studying': 1.0,
        'unemployed/sick leave/pension': 2.0
    }
}

top_path = rf'{os.getcwd()}'  # raw strings allow consistent slashes
ic(top_path)

# datapath = filedialog.askopenfile("Locate top level folder containing activity data")
data_path = top_path + r'/data'

def interpret_values(row, conversions, float_conv=0):
    for category in conversions:
        if category in row:
            if category == 'age':
                row[category] = str(conversions[category].get(row[category], row[category]))
            elif float_conv == 1:
                row[category] = float(conversions[category].get(row[category], row[category]))
            else:
                row[category] = conversions[category].get(row[category], row[category])
    return row

condition_numbers: list = []

def load_condition_data(scores_data_interpreted):
    numbers = scores_data_interpreted['number']
    global condition_numbers  # call and edit the global variable
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

        # ic(activity_data_temp.head())
        condition[num] = new_activity_data_temp

    # ic(condition)
    return condition

def load_scores() -> DF:
    scores_path = data_path + r'/scores.csv'
    scores_data = pd.read_csv(scores_path)
    ic(scores_data.head())  # check first few records in terminal output

    scores_data_interpreted = DF()
    for i, row in scores_data.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation)
        # ic(row_interpreted)
        scores_data_interpreted = pd.concat([scores_data_interpreted, row_interpreted], axis=1)

    scores_data_interpreted = scores_data_interpreted.T  # transpose

    # ic(scores_data_interpreted.head())
    return scores_data_interpreted

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

def scale_and_prepare(scores: DF = None, condition: dict = None):
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
    # ic(scores.head())

    demographic_temp: DF = scores[scores['number'].str.startswith('condition')]
    # ic(demographic_temp.head())

    demographic = DF()
    for i, field in enumerate(key_predictors):
        demographic = pd.concat([demographic, demographic_temp[field]], axis=1)

    # filter out all rows and columns except number condition_n, age, gender, madrs1, madrs2
    ic(demographic.head())

    demographic_encoded = DF()  # re-encode data back to integers
    for i, row in demographic.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation_reversed, float_conv=1)
        # print(f"{row=}, {row_interpreted=}")
        demographic_encoded = pd.concat([demographic_encoded, row_interpreted], axis=1)
    demographic_encoded = demographic_encoded.T

    ic(demographic_encoded.head())
    time.sleep(5)

    return patient_scaled_data, demographic_encoded, X_time_series

def build_LSTM(time_series_shape, supplementary_shape):
    # Define LSTM branch for time series data
    time_series_input = Input(shape=time_series_shape, name='time_series_input')
    x1 = LSTM(64)(time_series_input)

    # Define dense branch for supplementary data
    supplementary_input = Input(shape=supplementary_shape, name='supplementary_input')
    x2 = Dense(64, activation='relu')(supplementary_input)

    # Merge branches
    merged = concatenate([x1, x2])
    output = Dense(1, activation='linear')(merged)

    # Define and compile model
    model = Model(inputs=[time_series_input, supplementary_input], outputs=output)
    model.compile(optimizer=Adam(), loss='mse')

    return model

def train_LSTM(scores: pd.DataFrame = None, condition: dict = None):
    patient_scaled_activity_dict, demographic_refined, X_time_series = scale_and_prepare(scores=scores, condition=condition)

    X_time_series_combined = []
    X_supplementary_combined = []
    y_combined = []

    for patient_id, sequences in X_time_series.items():
        demographic_data = demographic_refined.loc[demographic_refined['number'] == patient_id].drop(columns=['number']).values
        if len(demographic_data) == 0:
            continue

        demographic_data_repeated = np.repeat(demographic_data, sequences.shape[0], axis=0)
        X_time_series_combined.append(sequences)
        X_supplementary_combined.append(demographic_data_repeated)

        y = demographic_refined.loc[demographic_refined['number'] == patient_id, 'madrs2'].values
        if len(y) > 0:
            y_combined.extend([y[0]] * sequences.shape[0])

    X_time_series_combined = np.concatenate(X_time_series_combined, axis=0).astype('float32')
    X_supplementary_combined = np.concatenate(X_supplementary_combined, axis=0).astype('float32')
    y_combined = np.array(y_combined, dtype=np.float32)

    model = build_LSTM(time_series_shape=(X_time_series_combined.shape[1], X_time_series_combined.shape[2]),
                             supplementary_shape=(X_supplementary_combined.shape[1],))

    history = model.fit([X_time_series_combined, X_supplementary_combined], y_combined, epochs=1, batch_size=32, validation_split=0.2)
    model.save('lstm_with_supplementary_data.h5')

    # Predict on validation data
    val_data = history.validation_data
    y_pred = model.predict([val_data[0][0], val_data[0][1]])
    y_true = val_data[1]

    # Calculate F1 Score
    y_pred_classes = np.round(y_pred)
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print(f'F1 Score: {f1}')

    return model

if __name__ == '__main__':
    scores_df = load_scores()  # dataframe of scores
    condition_dict_df = load_condition_data(scores_df)  # dict of key=condition_n, value=dataframe activity time series
    # cols = timestamp, time_since_start[mins], activity
    models = train_LSTM(scores=scores_df, condition=condition_dict_df)
