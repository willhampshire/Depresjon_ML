# Predicting Depression on MADRS scale based on time series Activity data using LSTM (RNN) machine learning

## Introduction
Using the ['Depresjon'](https://datasets.simula.no/depresjon/) open source dataset, an LSTM machine learning model was 
used to predict level of depression measured on the MADRS scale. There are 23 condition data files, where subjects were 
measured on the MADRS scale before and after the accompanying Activity time series measurements.

## Objective
The aim is to train a model capable of predicting, without clinical diagnosis, the MADRS score of a person by using
Activity data. The significance is that data of this nature could be easily accessible from pedometers and smart devices,
commonplace in society, given small adaptations. Small amounts of user-entered data on setup would facilitate this, 
labelled as supplementary data in this model. Scalability is promising for these reasons.

Used LSTM RNN from `tensorflow.keras` library, assisted by `scikit-learn` preprocessing, model selection and analysis 
using metrics to review the quality of the model (underfitted, overfitted etc.). Supplementary data allows good 
predictors for different categorisations of activity levels - for example, different age categories or genders may 
have different activity patterns.

The LSTM model was chosen as the Activity data series is a time series.
LSTM can effectively capture long-term dependencies and patterns in sequential data due to its ability to maintain 
information over extended periods, reducing the vanishing gradient problem some RNNs face. This makes 
LSTMs particularly well-suited for tasks where time context is crucial for accurate predictions.


## Training inputs & results
### Training Configuration
- **Epochs:** 10
- **Test split:** 30%
- **Batch size:** 32
- **Random state:** 42

### Model Performance
- **Mean Squared Error:** 5.147e-02
- **Root Mean Squared Error:** 2.269e-01
- **Mean Absolute Error:** 1.450e-01
- **R-squared:** 0.998

Anti-overfitting measures are implemented - for smaller datasets, it can be easy to train to specific data rather 
than learning for a general example (e.g. noise is fitted into the model); 
dropout layers, L2 regularization, and early stopping (not triggered).

Mean Absolute Error of 0.145 indicates the average absolute difference between predicted and actual values.
The value predicted is represented in the source data as 'madrs2'.


## Future considerations
- Train the model without initial MADRS score, eliminating the need for initial MADRS test.

