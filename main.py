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
    def on_train_begin(self, logs=None):
        # Initialize dictionaries to store losses for each output
        self.epoch_losses = DF()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs_df = DF([logs])

        self.epoch_losses = pd.concat([self.epoch_losses, logs_df])

    def on_train_end(self, logs=None):
        losses_path = os.path.join("results", f"epoch_losses.csv")
        self.epoch_losses.index = range(1, len(self.epoch_losses) + 1)
        self.epoch_losses = self.epoch_losses.reset_index(drop=False)
        self.epoch_losses.rename(columns={"index": "epoch"}, inplace=True)

        self.epoch_losses.to_csv(losses_path, index=False)


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
    demographic_input = Input(shape=(5,), name="demographic_input")

    # Define hyperparameters using Keras Tuner methods
    lstm_units = hp.Int("lstm_units", min_value=50, max_value=200, step=50)
    lstm_dropout = hp.Float("lstm_dropout", min_value=0.0, max_value=0.5, step=0.1)
    lstm_recurrent_dropout = hp.Float(
        "lstm_recurrent_dropout", min_value=0.0, max_value=0.5, step=0.1
    )
    l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01)
    dense_dem_units = hp.Int("dense_dem_units", min_value=32, max_value=64, step=32)
    dense1_units = hp.Int("dense1_units", min_value=32, max_value=128, step=32)
    dense1_dropout = hp.Float("dense1_dropout", min_value=0.0, max_value=0.5, step=0.1)

    lstm_layer = LSTM(
        lstm_units,
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
    )(time_series_input)

    dense_layer_dem = Dense(dense_dem_units, activation="relu")(demographic_input)

    combined_input = concatenate([lstm_layer, dense_layer_dem])

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
        max_trials=5,  # Number of trials - recc. 5
        executions_per_trial=3,  # Number of executions per trial - recc. 3
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

    X_train, X_test, y_train, y_test, dem_train, dem_test = train_test_split(
        X_combined, y_combined, demographic_refined, test_size=0.3, random_state=42
    )

    ic(
        1,
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
        dem_train.shape,
        dem_test.shape,
    )

    # Hyperparameter tuning
    tuner.search(
        [X_train, dem_train],
        [y_train[:, 0], y_train[:, 1]],
        epochs=5,  # adjust here - recc. 5
        validation_split=0.3,
        callbacks=[early_stopping, history],
    )

    # Retrieve best hyperparameters and build the best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Train the model with best hyperparameters
    best_model.fit(
        [X_train, dem_train],
        [y_train[:, 0], y_train[:, 1]],
        epochs=20,  # adjust here - recc. 20
        validation_split=0.3,
        callbacks=[early_stopping, history],
    )

    # Evaluate the model
    loss = best_model.evaluate(
        [X_test, dem_test],
        [y_test[:, 0], y_test[:, 1]],
    )

    # Predict on the test set
    y_pred = best_model.predict([X_test, dem_test])

    # Calculate R-squared for both outputs
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    r2_madrs2 = calculate_r2(y_test[:, 0], y_pred[0].flatten())
    r2_delta_madrs = calculate_r2(y_test[:, 1], y_pred[1].flatten())

    ic(loss)
    # Collect all metrics
    metrics = {
        "combined_loss": loss[0],  # Combined loss for both outputs
        "madrs2_loss": loss[1],  # Loss (MSE) for 'madrs2'
        "deltamadrs_loss": loss[2],  # Loss (MSE) for 'deltamadrs'
        "r2_madrs2": r2_madrs2,
        "r2_delta": r2_delta_madrs,
    }

    # Save the model
    model_save_path = os.path.join(
        top_path, "saved_models", model_save_name + f"_{timestamp}.keras"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path)

    # Save metrics to a JSON file
    with open(os.path.join("results", f"{model_save_name}_metrics.json"), "w") as f:
        json.dump(metrics, f)

    return best_model


timestamp = None


def main():
    """Main function to execute the training."""
    global timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    scores_data = load_scores()
    condition_data = load_condition_data(scores_data)

    model = train_and_save_model(
        scores=scores_data,
        condition=condition_data,
        model_save_name="lstm_2_targets",
    )


if __name__ == "__main__":
    main()
