import os
import time
import json
from icecream import ic
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
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


def scale_and_prepare(scores: DF = None, condition: dict = None):
    """Scale and prepare data for model training."""
    global sequence_length
    scalers = {}
    patient_scaled_data = {}
    X_time_series = {}

    for patient_id, patient_df in condition.items():
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(
            np.array(patient_df["activity"]).reshape(-1, 1)
        )
        scalers[patient_id] = scaler
        patient_scaled_data[patient_id] = scaled_data

    max_length = max(len(scaled_data) for scaled_data in patient_scaled_data.values())

    for patient_id, scaled_data in patient_scaled_data.items():
        X_time_series[patient_id] = create_sequences(scaled_data, sequence_length)

    key_predictors = ["number", "age", "gender", "madrs1", "madrs2"]
    add_predictor = "deltamadrs"

    demographic_temp = scores[scores["number"].str.startswith("condition")]
    demographic_temp.insert(
        len(key_predictors),
        add_predictor,
        demographic_temp["madrs2"] - demographic_temp["madrs1"],
    )

    key_predictors.append(add_predictor)
    demographic = demographic_temp[key_predictors]

    demographic_encoded = DF()
    for i, row in demographic.iterrows():
        row_interpreted = interpret_values(
            row, dataset_interpretation_reversed, float_conv=1
        )
        demographic_encoded = pd.concat([demographic_encoded, row_interpreted], axis=1)
    demographic_encoded = demographic_encoded.T

    return patient_scaled_data, demographic_encoded, X_time_series


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


def run_hyperparameter_tuning() -> kt.Hyperband:
    """Set up hyperparameter tuning using Keras Tuner."""
    tuner = kt.Hyperband(
        build_lstm_model,
        objective="val_loss",
        max_epochs=4,
        factor=3,
        directory="tuning_dir",
        project_name="LSTM_tuning",
    )
    return tuner


def train(
    scores: DF = None, condition: dict = None, model_save_name: str = "default_name"
) -> dict:
    """Train the LSTM model for each patient's data."""
    patient_scaled_data, demographic_refined, X_time_series = scale_and_prepare(
        scores=scores, condition=condition
    )

    model_store = {}
    losses_store = {}

    for patient_id, sequences in X_time_series.items():
        vprint(f"Training model for patient: {patient_id}")

        demographic_data = (
            demographic_refined.loc[demographic_refined["number"] == patient_id]
            .drop(columns=["number"])
            .to_numpy()
            .astype(np.float32)
        )

        # Preparing X and y
        X = sequences
        y_madrs2 = demographic_refined[demographic_refined["number"] == patient_id][
            "madrs2"
        ].values
        y_delta_madrs = demographic_refined[
            demographic_refined["number"] == patient_id
        ]["deltamadrs"].values

        y_madrs2 = np.repeat(y_madrs2, len(X)).astype(float)
        y_delta_madrs = np.repeat(y_delta_madrs, len(X)).astype(float)

        ic(X.shape, y_madrs2.shape, y_delta_madrs.shape)

        # Ensure y_madrs2 and y_delta_madrs have the same length as X
        if len(X) != len(y_madrs2):
            raise ValueError(
                f"Length mismatch between X and y arrays for patient {patient_id}"
            )

        # Create y_train and y_test arrays
        y = np.vstack([y_madrs2, y_delta_madrs]).T  # Shape: (num_samples, 2)
        y_train, y_test = train_test_split(y, test_size=0.3, random_state=42)

        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

        tuner = run_hyperparameter_tuning()

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = LossHistory()

        # Ensure y_train and y_test have the right dimensions
        y_train_0 = y_train[:, 0]
        y_train_1 = y_train[:, 1]
        y_test_0 = y_test[:, 0]
        y_test_1 = y_test[:, 1]

        tuner.search(
            [X_train, np.tile(demographic_data, (X_train.shape[0], 1))],
            [y_train_0, y_train_1],
            epochs=1,  # adjust here
            validation_split=0.3,
            callbacks=[early_stopping, history],
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(
            [X_train, np.tile(demographic_data, (X_train.shape[0], 1))],
            [y_train_0, y_train_1],
            epochs=2,  # adjust here
            validation_split=0.2,
            callbacks=[early_stopping, history],
        )

        loss = best_model.evaluate(
            [X_test, np.tile(demographic_data, (X_test.shape[0], 1))],
            [y_test_0, y_test_1],
        )

        model_store[patient_id] = best_model
        losses_store[patient_id] = loss

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(
        top_path, "saved_models", model_save_name + f"_{timestamp}"
    )
    os.makedirs(model_save_path, exist_ok=True)

    for patient_id, model in model_store.items():
        model.save(os.path.join(model_save_path, f"{model_save_name}_{patient_id}.h5"))

    with open(
        os.path.join(model_save_path, f"{model_save_name}_losses.json"), "w"
    ) as f:
        json.dump(losses_store, f)

    return model_store


def main():
    """Main function to execute the training."""
    scores_data = load_scores()
    condition_data = load_condition_data(scores_data)

    models = train(
        scores=scores_data,
        condition=condition_data,
        model_save_name="multi_patient_lstm",
    )


if __name__ == "__main__":
    main()
