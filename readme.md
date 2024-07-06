# Predicting Depression on MADRS scale based on time series Activity data using LSTM/RNN machine learning

## Introduction
Using the ['Depresjon'](https://datasets.simula.no/depresjon/) open source dataset, an LSTM machine learning model was 
used to predict level of depression measured on the MADRS scale. There are 23 condition data files, where subjects were 
measured on the MADRS scale before and after the accompanying Activity time series measurements.

## Objective
The aim is to train a model using the LSTM RNN from `tensorflow.keras` library, assisted by `scikit-learn` 
preprocessing, model selection and analysis using metrics to review the quality of the model (underfitted, overfitted etc.).

The LSTM model was chosen as the Activity data series is a time series.
LSTM can effectively capture long-term dependencies and patterns in sequential data due to its ability to maintain 
information over extended periods, reducing the vanishing gradient problem some RNNs face. This makes 
LSTMs particularly well-suited for tasks where time context is crucial for accurate predictions.


## Training inputs & results
Mean Squared Error: 5.147e-02
Root Mean Squared Error: 2.269e-01
Mean Absolute Error: 1.450e-01
R-squared: 0.998
Epochs: 10
Test split: 30%
Batch size: 32
Random state: 42
