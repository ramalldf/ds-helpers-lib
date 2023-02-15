from pkg_resources import resource_stream
from matplotlib import pyplot as plt
from matplotlib import rcParams

import pandas as pd

def load_classifier_predictions():
    '''Loads classifier prediction data'''

    stream_object = resource_stream('ds_helpers', 'data/example_classifier_predictions.csv')

    # Read table into pandas dataframe
    temp_df = pd.read_csv(stream_object, index_col=0)

    return temp_df

def load_mpl_style():
    '''Updates plotting style'''

    plt.style.use('fivethirtyeight')

    # Define dict
    new_dict = {'figure.facecolor': 'red'}

    # Update style
    rcParams.update(new_dict)
    
    # stream_object = resource_stream('ds_helpers', 'data/ds_helpers.mplstyle')

    # # Read table into pandas dataframe
    # plt.style.use(stream_object)
