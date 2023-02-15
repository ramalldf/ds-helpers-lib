from pkg_resources import resource_stream
from matplotlib import pyplot as plt
import pandas as pd

def load_classifier_predictions():
    '''Loads classifier prediction data'''

    stream_object = resource_stream('ds_helpers', 'data/example_classifier_predictions.csv')

    # Read table into pandas dataframe
    temp_df = pd.read_csv(stream_object, index_col=0)

    return temp_df

def load_mpl_style():
    '''Updates plotting style'''

    hosted_url = 'https://raw.githubusercontent.com/ramalldf/ds-helpers/update-plot-pairs/ds_helpers/data/stylesheet.mplstyle'

    # Load stylesheet
    #plt.style.use('ds_helpers.data.stylesheet')
    plt.style.use(hosted_url)
