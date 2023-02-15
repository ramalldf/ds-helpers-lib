from pkg_resources import resource_stream
from matplotlib import pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from ds_helpers.data.style_params import style_dict

import pandas as pd

def load_classifier_predictions():
    '''Loads classifier prediction data'''

    stream_object = resource_stream('ds_helpers', 'data/example_classifier_predictions.csv')

    # Read table into pandas dataframe
    temp_df = pd.read_csv(stream_object, index_col=0)

    return temp_df

def load_mpl_style():
    '''Updates plotting style'''

    # Update style using predefined values
    rcParams.update(style_dict)
