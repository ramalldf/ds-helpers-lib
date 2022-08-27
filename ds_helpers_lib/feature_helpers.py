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

# TODO: add  wrapper fxn 
def mean_encode_cat_feats(input_df, feature_columns, target_col, smoothing_param, model_object=None):
    '''Wrapper function to target encode a list of categorical feats'''

    temp_df = input_df.copy()

    # If no model given just map values
    if model_object:

        # Iterate through categorical columns and apply mapping
        for i in feature_columns:
            feature_encoding = model_object[i]['mean_encoding_map']
            temp_df[i] = temp_df[i].map(feature_encoding)

        # Return encoded df
        return temp_df
    
    # Else iteratively build encoding map and append to cat feat map    
    else:
        cat_feat_map = {}

        for j in feature_columns:
            me_object = mean_encode_feature(input_df, j, target_col, smoothing_param)
            me_map = me_object['mean_encoding_map']
            temp_df[j] = me_object['mean_encoded_values']

            # Update feature map with mappings for this feature
            cat_feat_map[i] = me_map
        
        return temp_df, cat_feat_map
        


# TODO: add linear dt fxn
