import numpy as np
import pandas as pd


def mean_encode_feature(input_df: pd.DataFrame, category: str, target: str, smoothing_param: int) -> pd.DataFrame:
    '''Returns a list of mean-encoded values'''

    # Init copy
    temp_df = input_df.copy()

    # Calculate global mean of target
    global_mean = temp_df[target].mean()

    # Calc. count and mean aggs 
    aggs = temp_df.groupby(category)[target].agg(['count', 'mean'])
    val_counts = aggs['count']
    val_means = aggs['mean'] 

    # Smooth mean values
    smoothed_vals = (val_counts * val_means + smoothing_param * global_mean) / (val_counts + smoothing_param)

    # Apply transformation and return
    return temp_df[category].map(smoothed_vals)