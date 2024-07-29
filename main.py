import os
import time
import json
from icecream import ic  # debugging output
import numpy as np
import pandas as pd
from pandas import DataFrame as DF  # create and typehint shorthand
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt

from debug_tools.print_verbose import print_verbose as vprint  # import custom print debugging tool

sequence_length = 10  # X time series sequence length

# interpretation of raw data for human analysis
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

# convert data to ML compatible data
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

top_path = rf'{os.getcwd()}'  # raw strings allow consistent path slashes
ic(top_path)

data_path = top_path + r'/data'  # if repo pulled fully

class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        # Initialize dictionaries to store losses for each output
        self.epoch_losses = {
            'madrs2_loss': [],
            'deltamadrs_loss': [],
            'val_madrs2_loss': [],
            'val_deltamadrs_loss': []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get training losses for each output
        if 'loss' in logs:
            # Assuming 'loss' is a list of losses for each output
            if isinstance(logs['loss'], list):
                self.epoch_losses['madrs2_loss'].append(logs['loss'][0])
                self.epoch_losses['deltamadrs_loss'].append(logs['loss'][1])
            else:
                self.epoch_losses['madrs2_loss'].append(-1)
                self.epoch_losses['deltamadrs_loss'].append(-1)

        # Get validation losses for each output
        if 'val_loss' in logs:
            # Assuming 'val_loss' is a list of validation losses for each output
            if isinstance(logs['val_loss'], list):
                self.epoch_losses['val_madrs2_loss'].append(logs['val_loss'][0])
                self.epoch_losses['val_deltamadrs_loss'].append(logs['val_loss'][1])
            else:
                self.epoch_losses['val_madrs2_loss'].append(-1)
                self.epoch_losses['val_deltamadrs_loss'].append(-1)


def loss_mse(y_true, y_pred):
    mse = MeanSquaredError()
    loss1 = mse(y_true[0], y_pred[0])
    loss2 = mse(y_true[1], y_pred[1])
    return loss1 + loss2

def interpret_values(row, conversions, float_conv=0) -> pd.Series:
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


def load_condition_data(scores_data_interpreted) -> dict:
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
                                                            new_activity_data_temp['timestamp'].iloc[
                                                                0]).dt.total_seconds() / 60.0
        new_activity_data_temp['activity'] = activity_data_temp['activity']

        condition[num] = new_activity_data_temp

    # ic(condition)
    return condition  # return explicitly despite global definition


def load_scores() -> DF:
    scores_path = data_path + r'/scores.csv'
    scores_data = pd.read_csv(scores_path)
    # ic(scores_data.head())  # check first few records in terminal output

    scores_data_interpreted = DF()
    for i, row in scores_data.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation)
        scores_data_interpreted = pd.concat([scores_data_interpreted, row_interpreted], axis=1)

    scores_data_interpreted = scores_data_interpreted.T  # transpose

    # ic(scores_data_interpreted.head())
    return scores_data_interpreted


def create_sequences(data, sequence_length, max_super_sequence_size, mask=-1) -> np.ndarray:
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)

    max_len = max_super_sequence_size - len(sequences)
    shape = ((0, max_len), (0, 0), (0, 0))
    # ic(max_super_sequence_size, shape, np.shape(sequences))
    finished_sequences = np.pad(sequences, shape, mode='constant', constant_values=-1)
    # ic(max_super_sequence_size, finished_sequences, finished_sequences.shape)
    return finished_sequences


def scale_and_prepare(scores: DF = None, condition: dict = None):
    """
    Scales data and prepares data format for model training.
    :param scores: DF
    :param condition: dict
    :return: patient_scaled_data dict, demographic_encoded DF, X_time_series np.ndarray
    """
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

    max_length: int = 0
    for _, scaled_data in patient_scaled_data.items():
        scaled_data_len: int = len(scaled_data)
        if int(scaled_data_len) > int(max_length):
            max_length = scaled_data_len

    for patient_id, scaled_data in patient_scaled_data.items():
        X_time_series[patient_id] = create_sequences(scaled_data, sequence_length, max_length)

    key_predictors = ['number', 'age', 'gender', 'madrs1', 'madrs2']
    add_predictor = 'deltamadrs'

    demographic_temp: DF = scores[scores['number'].str.startswith('condition')]
    # demographic_temp[add_predictor] = demographic_temp['madrs2'] - demographic_temp['madrs1']
    demographic_temp.insert(len(key_predictors), add_predictor, demographic_temp['madrs2'] - demographic_temp['madrs1'])

    key_predictors.append(add_predictor)

    demographic = DF()
    for i, field in enumerate(key_predictors):
        demographic = pd.concat([demographic, demographic_temp[field]], axis=1)

    # filter out all rows and columns except number condition_n, age, gender, madrs1, madrs2
    demographic_encoded = DF()  # re-encode data back to integers
    for i, row in demographic.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation_reversed, float_conv=1)
        # print(f"{row=}, {row_interpreted=}")
        demographic_encoded = pd.concat([demographic_encoded, row_interpreted], axis=1)
    demographic_encoded = demographic_encoded.T

    return patient_scaled_data, demographic_encoded, X_time_series


def build_LSTM(hp) -> Model:
    # Define LSTM branch for time series data with dropout and L2 regularization
    time_series_input = Input(shape=(None, 1), name='time_series_input')
    time_series_masked_input = Masking(mask_value=-1.0)(time_series_input)
    lstm_units = hp.get('lstm_units')
    l2_reg = hp.get('l2_reg')
    x1 = LSTM(lstm_units, kernel_regularizer=l2(l2_reg))(time_series_masked_input)
    dropout_rate = hp.get('dropout_rate')
    x1 = Dropout(dropout_rate)(x1)

    # Define dense branch for supplementary data with dropout and L2 regularization
    supplementary_input = Input(shape=(3,), name='supplementary_input')  # Fixed input shape of 3 demographic features
    dense_units = hp.get('dense_units')
    x2 = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg))(supplementary_input)
    x2 = Dropout(dropout_rate)(x2)

    # Merge branches
    merged = concatenate([x1, x2])
    dense_concatenated = Dense(32, activation='relu')(merged)
    madrs2 = Dense(1, activation='relu')(dense_concatenated)
    deltamadrs = Dense(1, activation='relu')(dense_concatenated)

    # Define and compile model
    learning_rate = hp.get('learning_rate')
    model = Model(inputs=[time_series_input, supplementary_input], outputs=[madrs2, deltamadrs])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model


def train_LSTM(scores: pd.DataFrame = None, condition: dict = None, model_save_name: str = 'default_name') -> Model:
    patient_scaled_activity_dict, demographic_refined, X_time_series = scale_and_prepare(scores=scores,
                                                                                         condition=condition)

    X_time_series_combined = []
    X_supplementary_combined = []
    y_combined = []

    for patient_id, sequences in X_time_series.items():

        demographic_data = demographic_refined.loc[demographic_refined['number'] == patient_id].drop(
            columns=['number', 'madrs2', 'deltamadrs']).values
        if len(demographic_data) == 0:
            continue

        # demographic_data_repeated = np.repeat(demographic_data, sequences.shape[0], axis=0)
        X_time_series_combined.append(sequences)
        X_supplementary_combined.append(demographic_data)

        y = demographic_refined.loc[demographic_refined['number'] == patient_id, ['madrs2', 'deltamadrs']].values
        if len(y) > 0:
            y_combined.append(y[0])

    x_series_shape = np.shape(X_time_series_combined)
    x_series_shape_new = (x_series_shape[0], x_series_shape[1]*x_series_shape[2], x_series_shape[3])
    X_time_series_combined = np.array(X_time_series_combined).astype(
        'float32').reshape(x_series_shape_new)
    X_supplementary_combined = np.concatenate(X_supplementary_combined, axis=0).astype('float32')
    y_combined = np.array(y_combined, dtype=np.float32)

    print(
        f"Shape should be (n, 2) {y_combined.shape}\n(n, 3) {X_supplementary_combined.shape}"
        f"\n(n, x * w, 1) {X_time_series_combined.shape} - often (23, x * 10, 1)")

    ic(X_time_series_combined.shape)

    # Hyperparameter tuner setup
    hp = kt.HyperParameters()
    hp.Int('lstm_units', min_value=64, max_value=128, step=8)
    hp.Float('l2_reg', min_value=0.01, max_value=0.1, step=0.01)
    hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    hp.Int('dense_units', min_value=64, max_value=128, step=8)
    hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    tuner = kt.RandomSearch(
        build_LSTM,
        objective='val_loss',
        max_trials=10,  # set low for debugging
        executions_per_trial=2,
        directory='hp_tuning',
        project_name='lstm_dual_output',
        hyperparameters=hp,
        tune_new_entries=False
    )

    y1, y2 = np.split(y_combined, 2, axis=1)  # madrs2, dmadrs
    # Split data into training and validation sets
    X_train_sup, X_val_sup, y_madrs, y_madrs_test, y_d_madrs, y_d_madrs_test = \
        train_test_split(
            X_supplementary_combined,
            y1,
            y2,
            test_size=0.3,
            random_state=42)

    X_train_ts, X_val_ts = train_test_split(
        X_time_series_combined,
        test_size=0.3,
        random_state=42)

    ic(X_train_ts.shape)


    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # likely will not activate unless epoch number increased from 10
    losses = LossHistory()

    tuner.search([X_train_ts, X_train_sup], [y_madrs, y_d_madrs], epochs=2, batch_size=32,
                 validation_data=([X_val_ts, X_val_sup], [y_madrs_test, y_d_madrs_test]), callbacks=[early_stopping])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Optimal number of LSTM units: {best_hps.get('lstm_units')}")
    print(f"Optimal L2 regularization: {best_hps.get('l2_reg')}")
    print(f"Optimal dropout rate: {best_hps.get('dropout_rate')}")
    print(f"Optimal number of Dense units: {best_hps.get('dense_units')}")
    print(f"Optimal learning rate: {best_hps.get('learning_rate')}")

    # Build the model with the optimal hyperparameters and train it
    model = tuner.hypermodel.build(best_hps)

    model_fitted = model.fit([X_train_ts, X_train_sup], [y_madrs, y_d_madrs], epochs=50, batch_size=32,
                             validation_data=([X_val_ts, X_val_sup], [y_madrs_test, y_d_madrs_test]),
                             callbacks=[early_stopping, losses])
    model.save(model_save_name)
    ic(losses.epoch_losses)
    # Predict on validation data
    pred_madrs, pred_d_madrs = model.predict([X_val_ts, X_val_sup])  # madrs2

    model_results = {}
    model_shapes = {"X Time": X_time_series_combined.shape,
                    "X Supp": X_supplementary_combined.shape,
                    "Y madrs": y1.shape,
                    "Y d_madrs": y2.shape}

    model_results["Model used"] = str(model_save_name)
    model_results["Original Y data"] = np.array([y_madrs, y_d_madrs]).astype(float)
    model_results["Predicted Y data"] = np.array([pred_madrs, pred_d_madrs]).astype(float)
    model_results["Data shapes"] = model_shapes

    cwd = fr'{os.getcwd()}'
    model_results_fname = cwd + r"/results/results_for_eval.json"
    with open(model_results_fname, 'w') as json_file:
        json.dump(model_results, json_file, indent=4)

    return model, losses


if __name__ == '__main__':
    scores_df = load_scores()  # dataframe of scores
    condition_dict_df = load_condition_data(scores_df)  # dict of key=condition_n, value=dataframe activity time series
    # cols = timestamp, time_since_start[mins], activity

    models, losses = train_LSTM(scores=scores_df, condition=condition_dict_df, model_save_name='depresjon_2.keras')
