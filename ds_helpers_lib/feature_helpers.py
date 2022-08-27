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

    # Smooth mean values and apply to encode raw values
    smoothed_map = (val_counts * val_means + smoothing_param * global_mean) / (val_counts + smoothing_param)
    encoded_values = temp_df[category].map(smoothed_vals)

    # Add map and encoding to dictionary
    output = {
              'mean_encoded_values': encoded_values,
              'mean_encoding_map': smoothed_map
             }  

    # Apply transformation and return
    return output
