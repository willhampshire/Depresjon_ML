# Predicting Depression on MADRS Scale using LSTM (RNN) Machine Learning with Motor Activity Time Series Data

## Introduction
Using the ['Depresjon'](https://datasets.simula.no/depresjon/) open source dataset, an LSTM machine learning model was 
used to predict level of depression measured on the MADRS scale. There are 23 condition data files, where subjects were 
measured on the MADRS scale before and after the accompanying Activity time series measurements. Data is fed into the model
in two parts; time series and additional demographic data.

## Objective
The aim is to train a model capable of predicting, without clinical diagnosis, the MADRS score of a person by using
Activity data. The significance is that data of this nature could be easily accessible from pedometers and smart devices,
commonplace in society, given small adaptations. Small amounts of user-entered data on setup would facilitate this, 
labelled as supplementary data in this model. Scalability is promising for these reasons.

Used LSTM RNN from `tensorflow.keras` library, assisted by `scikit-learn` preprocessing, model selection and analysis 
using metrics to review the quality of the model (underfitted, overfitted etc.). Supplementary data allows good 
predictors for different categorisations of activity levels - for example, different age categories or genders may 
have different activity patterns.

The LSTM model was chosen, as the Activity data series is a time series.
LSTM can effectively capture long-term dependencies and patterns in sequential data due to its ability to maintain 
information over extended periods, reducing the vanishing gradient problem some RNNs face. This makes 
LSTMs particularly well-suited for tasks where time context is crucial for accurate predictions.

## Data Source and Preprocessing
### Data Source
The ['Depresjon'](https://datasets.simula.no/depresjon/) dataset consists of 23 condition files of activity data with associated demographic data & MADRS scores. It is published 
as pre-anonymised open source data so that researchers can use the data to develop models such as this, to advance research &ndash; 
"the available data may eventually help researchers to develop systems capable of automatically detecting depression 
states based on sensor data".

"[The dataset was originally collected for the study of motor activity in schizophrenia and major depression (doi.org/10.1186/1756-0500-3-149)](https://bmcresnotes.biomedcentral.com/articles/10.1186/1756-0500-3-149). Motor activity was monitored with an actigraph watch worn at the right wrist (Actiwatch, Cambridge Neurotechnology Ltd, England, model AW4). The actigraph watch measures activity levels. The sampling frequency is 32Hz and movements over 0.05 g are recorded. A corresponding voltage is produced and is stored as an activity count in the memory unit of the actigraph watch. The number of counts is proportional to the intensity of the movement. Total activity counts were continuously recorded in one minute intervals."

### Data Preprocessing
  - Activity data is normalized using MinMax scaling.
  - Sequential data is segmented into fixed-length sequences suitable for LSTM input.
  - Missing data and outliers are handled through appropriate preprocessing techniques.


## Model & Training
### Architecture
- **Model Architecture:** The LSTM model architecture comprises:
  - LSTM layers to capture temporal dependencies in the activity data.
  - Dense layers for supplementary demographic data.
  - Dropout layers and L2 regularization to prevent overfitting.
- **Training Process Configuration:** 
  - Optimizer: Adam optimizer with default settings.
  - Realtime Displayed Loss Function: Mean Squared Error ('MSE').

### Training Input Configuration
- **Epochs:** 10
- **Test split:** 30%
- **Batch size:** 32
- **Random state:** 42
- **Raw Data Samples:** 23

### Performance Metrics
- **Mean Squared Error:** 5.147e-02
- **Root Mean Squared Error:** 2.269e-01
- **Mean Absolute Error:** 1.450e-01
- **R-squared:** 0.998

Anti-overfitting measures - dropout layers, L2 regularization, and early stopping (not triggered) - are implemented. 
For smaller datasets, it can be easy to train to specific data rather than learning more holistically (e.g. noise 
is fitted into the model).

Mean Absolute Error (MAE) of 0.145 indicates the average absolute difference between predicted and actual values.
The value predicted is represented in the source data as 'madrs2', and the context of this number means anything of the 
order <1 means an accurate prediction is made. Additionally, 99.8% of the variance in the prediction is explained within
the model (R-squared) - this indicates the data is fitted well. Samples tested on: 23 × 30% = 6.9 => 7.

Relative MAE = MAE / MADRS range = (0.145 / 60) × 100 = 0.242%


## Future considerations
- **Initial data dependency:** Train the model without initial MADRS score, eliminating the need for initial MADRS test.
- **Model Generalization:** Challenges and considerations for deploying the model include adapting to diverse activity data sources and generalizing across different demographics. For example, currently, Min age = 20, Max age = 69, in bins of 5yrs.
- **Neural Layers Improvement:** Future enhancements may involve exploring alternative neural network architectures / adding more layers, and incorporating more sophisticated preprocessing techniques.

## Acknowledgments
- **Dataset Attribution:** The ['Depresjon'](https://datasets.simula.no/depresjon/) dataset is sourced from [Simula](https://datasets.simula.no/). Proper attribution is given to the dataset creators and contributors.
- **Tool Acknowledgment:** The project utilizes TensorFlow-Keras for deep learning and scikit-learn for preprocessing and evaluation metrics.
