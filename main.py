import os
import time
import json
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

from debug_tools.print_verbose import print_verbose as vprint
from icecream import ic

# tf.debugging.set_log_device_placement(True)

# Define constants and dataset mappings
sequence_length = 10  # Window size for time series data

# Interpretation of raw data for human analysis
dataset_interpretation = {
    "gender": {1: "female", 2: "male"},
    "afftype": {1: "bipolar II", 2: "unipolar depressive", 3: "bipolar I"},
    "melanch": {1: "melancholia", 2: "no melancholia"},
    "inpatient": {1: "inpatient", 2: "outpatient"},
    "marriage": {1: "married or cohabiting", 2: "single"},
    "work": {1: "working or studying", 2: "unemployed/sick leave/pension"},
}

# Reversed mappings for encoding demographic data
dataset_interpretation_reversed = {
    "age": {
        "20-24": 1.0,
        "25-29": 2.0,
        "30-34": 3.0,
        "35-39": 4.0,
        "40-44": 5.0,
        "45-49": 6.0,
        "50-54": 7.0,
        "55-59": 8.0,
        "60-64": 9.0,
        "65-69": 10.0,
    },
    "gender": {"female": 1.0, "male": 2.0},
    "work": {"working or studying": 1.0, "unemployed/sick leave/pension": 2.0},
}

top_path = rf"{os.getcwd()}"
ic(top_path)

data_path = top_path + r"/data"  # Adjust to your data directory


class LossHistory(Callback):
    """Custom callback to record losses during training."""

    def on_train_begin(self, logs=None):
        self.epoch_losses = {
            "madrs2_loss": [],
            "deltamadrs_loss": [],
            "val_madrs2_loss": [],
            "val_deltamadrs_loss": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "loss" in logs:
            if isinstance(logs["loss"], list):
                self.epoch_losses["madrs2_loss"].append(logs["loss"][0])
                self.epoch_losses["deltamadrs_loss"].append(logs["loss"][1])
            else:
                self.epoch_losses["madrs2_loss"].append(-1)
                self.epoch_losses["deltamadrs_loss"].append(-1)
        if "val_loss" in logs:
            if isinstance(logs["val_loss"], list):
                self.epoch_losses["val_madrs2_loss"].append(logs["val_loss"][0])
                self.epoch_losses["val_deltamadrs_loss"].append(logs["val_loss"][1])
            else:
                self.epoch_losses["val_madrs2_loss"].append(-1)
                self.epoch_losses["val_deltamadrs_loss"].append(-1)


def interpret_values(row, conversions, float_conv=0) -> pd.Series:
    """Convert dataset row values based on provided mappings."""
    for category in conversions:
        if category in row:
            if category == "age":
                row[category] = str(
                    conversions[category].get(row[category], row[category])
                )
            elif float_conv == 1:
                row[category] = float(
                    conversions[category].get(row[category], row[category])
                )
            else:
                row[category] = conversions[category].get(row[category], row[category])
    return row


def load_condition_data(scores_data_interpreted) -> dict:
    """Load time series data for each condition (patient) from CSV files."""
    numbers = scores_data_interpreted["number"]
    condition_numbers = [item for item in numbers if not item.startswith("control")]
    condition_data = {}
    condition_path = data_path + r"/condition"

    for num in condition_numbers:
        condition_path_num = condition_path + rf"/{num}.csv"
        activity_data_temp = pd.read_csv(condition_path_num)
        new_activity_data_temp = DF()

        new_activity_data_temp["timestamp"] = pd.to_datetime(
            activity_data_temp["timestamp"]
        )
        new_activity_data_temp["time_since_start[mins]"] = (
            new_activity_data_temp["timestamp"]
            - new_activity_data_temp["timestamp"].iloc[0]
        ).dt.total_seconds() / 60.0
        new_activity_data_temp["activity"] = activity_data_temp["activity"]

        condition_data[num] = new_activity_data_temp

    return condition_data


def load_scores() -> DF:
    """Load and interpret scores from the scores.csv file."""
    scores_path = data_path + r"/scores.csv"
    scores_data = pd.read_csv(scores_path)
    scores_data_interpreted = DF()

    for i, row in scores_data.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation)
        scores_data_interpreted = pd.concat(
            [scores_data_interpreted, row_interpreted], axis=1
        )

    scores_data_interpreted = scores_data_interpreted.T
    return scores_data_interpreted


def create_sequences(data, sequence_length) -> np.ndarray:
    """Create sliding window sequences from time series data."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i : i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


def scale_and_prepare(
    scores: pd.DataFrame = None, condition: Dict[str, pd.DataFrame] = None
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Scale and prepare data for model training."""
    global sequence_length
    scalers = {}
    patient_scaled_data = {}
    X_time_series = {}

    # Scale data for each patient
    for patient_id, patient_df in condition.items():
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(
            np.array(patient_df["activity"])
            .reshape(-1, 1)
            .astype(float)  # Ensure float type
        )
        scalers[patient_id] = scaler
        patient_scaled_data[patient_id] = scaled_data

    # Create sequences for each patient
    for patient_id, scaled_data in patient_scaled_data.items():
        X_time_series[patient_id] = create_sequences(scaled_data, sequence_length)

    key_predictors = ["number", "age", "gender", "madrs1", "madrs2"]
    add_predictor = "deltamadrs"

    # Prepare demographic data
    demographic_temp = scores[scores["number"].str.startswith("condition")].copy()
    demographic_temp[add_predictor] = demographic_temp["madrs2"].astype(
        float
    ) - demographic_temp["madrs1"].astype(float)

    key_predictors.append(add_predictor)
    demographic = demographic_temp[key_predictors]

    # Encode the demographic data
    for column in demographic.columns:
        if column in dataset_interpretation_reversed:
            demographic.loc[:, column] = demographic[column].map(
                dataset_interpretation_reversed[column]
            )

    # Combine all time series data and demographic data for all patients
    all_sequences = []
    all_y_madrs2 = []
    all_y_delta_madrs = []
    all_demographics = []

    for patient_id, sequences in X_time_series.items():
        num_sequences = len(sequences)
        all_sequences.append(sequences)

        # Extract the demographic row for the current patient
        patient_demographic = demographic[demographic["number"] == patient_id]

        # Repeat the demographic data row to match the number of sequences
        repeated_demographic = pd.DataFrame(
            np.repeat(patient_demographic.values, num_sequences, axis=0),
            columns=demographic.columns,
        )
        all_demographics.append(repeated_demographic)

        y_madrs2 = np.repeat(patient_demographic["madrs2"].values, num_sequences)
        y_delta_madrs = np.repeat(
            patient_demographic["deltamadrs"].values, num_sequences
        )
        all_y_madrs2.append(y_madrs2)
        all_y_delta_madrs.append(y_delta_madrs)

    X_combined = np.vstack(all_sequences).astype(float)  # Ensure float type
    y_madrs2_combined = np.concatenate(all_y_madrs2).astype(float)
    y_delta_madrs_combined = np.concatenate(all_y_delta_madrs).astype(float)

    y_combined = np.vstack([y_madrs2_combined, y_delta_madrs_combined]).T.astype(float)

    # Concatenate all repeated demographic data frames into one
    demographic_combined = pd.concat(all_demographics, ignore_index=True)

    return X_combined, demographic_combined, y_combined


def build_lstm_model(hp) -> Model:
    """Build the LSTM model for training."""
    time_series_input = Input(shape=(None, 1), name="time_series_input")
    time_series_masked_input = Masking(mask_value=-1.0)(time_series_input)

    # Define hyperparameters using Keras Tuner methods
    lstm_units = hp.Int("lstm_units", min_value=50, max_value=200, step=50)
    lstm_dropout = hp.Float("lstm_dropout", min_value=0.0, max_value=0.5, step=0.1)
    lstm_recurrent_dropout = hp.Float(
        "lstm_recurrent_dropout", min_value=0.0, max_value=0.5, step=0.1
    )
    l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01)

    lstm_layer = LSTM(
        lstm_units,
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
    )(time_series_masked_input)

    demographic_input = Input(shape=(5,), name="demographic_input")

    combined_input = concatenate([lstm_layer, demographic_input])

    dense1_units = hp.Int("dense1_units", min_value=32, max_value=128, step=32)
    dense1_dropout = hp.Float("dense1_dropout", min_value=0.0, max_value=0.5, step=0.1)

    dense_layer = Dense(dense1_units, activation="relu", kernel_regularizer=l2(l2_reg))(
        combined_input
    )
    dropout_layer = Dropout(dense1_dropout)(dense_layer)

    output1 = Dense(1, name="madrs2")(dropout_layer)
    output2 = Dense(1, name="deltamadrs")(dropout_layer)

    model = Model(
        inputs=[time_series_input, demographic_input], outputs=[output1, output2]
    )
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=["mae", "mae"])

    return model


def run_hyperparameter_tuning() -> kt.RandomSearch:
    """Set up hyperparameter tuning using Keras Tuner."""
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective="val_loss",
        max_trials=2,  # Number of trials
        executions_per_trial=1,  # Number of executions per trial
        directory="hp_tuning",
        project_name="LSTM_tuning",
    )
    return tuner


def train_and_save_model(
    scores: DF = None, condition: dict = None, model_save_name: str = "default_name"
) -> dict:
    """Train the LSTM model and save it as a single .keras file."""
    X_combined, demographic_refined, y_combined = scale_and_prepare(
        scores=scores, condition=condition
    )
    demographic_refined = demographic_refined.drop("number", axis=1)
    demographic_refined = demographic_refined.to_numpy().astype(float)

    tuner = run_hyperparameter_tuning()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = LossHistory()

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.3, random_state=42
    )

    ic(1, X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    tuner.search(
        [X_train, demographic_refined],
        [y_train[:, 0], y_train[:, 1]],
        epochs=1,  # adjust here
        validation_split=0.3,
        callbacks=[early_stopping, history],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(
        [
            X_train,
            demographic_refined,
        ],
        [y_train[:, 0], y_train[:, 1]],
        epochs=2,  # adjust here
        validation_split=0.2,
        callbacks=[early_stopping, history],
    )

    loss = best_model.evaluate(
        [X_test, demographic_refined],
        [y_test[:, 0], y_test[:, 1]],
    )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(
        top_path, "saved_models", model_save_name + f"_{timestamp}.keras"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_model.save(model_save_path)

    with open(os.path.join("results", f"{model_save_name}_losses.json"), "w") as f:
        json.dump(loss, f)

    return best_model


def main():
    """Main function to execute the training."""
    scores_data = load_scores()
    condition_data = load_condition_data(scores_data)

    model = train_and_save_model(
        scores=scores_data,
        condition=condition_data,
        model_save_name="combined_lstm_model",
    )


if __name__ == "__main__":
    main()
