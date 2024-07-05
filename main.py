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


def load_activity_data():
    data = []

def load_scores():
    scores_path = data_path + r'/scores.csv'
    scores_data = pd.read_csv(scores_path)
    ic(scores_data.head()) # check first few records in terminal output

    scores_data_interpreted = DF()
    for i,row in scores_data.iterrows():
        row_interpreted = interpret_values(row, dataset_interpretation)
        ic(row_interpreted)
        scores_data_interpreted = pd.concat([scores_data_interpreted, row_interpreted], axis=1)

    scores_data_interpreted = scores_data_interpreted.T #transpose
    ic(scores_data_interpreted.head())




if __name__ == '__main__':
    load_scores()
    #load_activity_data()

