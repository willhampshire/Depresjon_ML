import os
from tkinter import filedialog
from icecream import ic
import pandas as pd
from pandas import DataFrame as DF #often used so shortened alias

dataset_interpretation: dict = {'gender': {
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


top_path = rf'{os.getcwd()}' # raw strings allow consistent slashes
ic(top_path)

#datapath = filedialog.askopenfile("Locate top level folder containing activity data")
data_path = top_path + r'/data'

def interpret_values(row, conversions):
    for category in conversions:
        if category in row:
            row[category] = conversions[category].get(row[category], row[category])
    return row


def load_condition_data(scores_data_interpreted):
    numbers = scores_data_interpreted['number']
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

    ic(condition)
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
    ic(scores_data_interpreted.head())
    return scores_data_interpreted

def train_LSTM():
    pass




if __name__ == '__main__':
    scores_df = load_scores() #dataframe of scores
    condition_dict_df = load_condition_data(scores_df) #dict of key=condition_n, value=dataframe activity time series
        #cols = timestamp, time_since_start[mins], activity
    train_LSTM()
