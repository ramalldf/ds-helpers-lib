import numpy as np
import pandas as pd


class MeanEncoder:
    def __init__(self, smoothing_param):
        self.smoothing_param = smoothing_param    
    
    
    def me_calculate(self, input_df, category, target, smoothing_param):
        '''Returns a list of mean-encoded values. Smoothing param
        defines the number of instances of a subgroup in the category
        that must exist before that groups mean overcomes the global mean'''

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
        encoded_values = temp_df[category].map(smoothed_map)

        # Add map and encoding to dictionary
        output = {
                  'mean_encoded_values': encoded_values,
                  'mean_encoding_map': smoothed_map
                 }  

        # Apply transformation and return
        return output

    def me_calculate_features(self, input_df, features_list, target_col):
        '''Iterates through each feature column to build a reference dict that
        has the mean encoding map for all features'''

        # Initialize master encoding map and temporary df
        master_encoding_map = {}
        temp_df = input_df.copy()

        # Calculate the mean encoding values for every feature, update values in columns
        # and updaet master map with encodings
        for j in features_list:
            me_object = self.me_calculate(temp_df, j, target_col, self.smoothing_param)
            temp_map = me_object['mean_encoding_map']
            temp_df[j] = me_object['mean_encoded_values']

            # Update feature map with mappings for this feature
            master_encoding_map[j] = temp_map
        
        # Assign final map as an attribute
        self.master_encoding_map = master_encoding_map


    def me_encode_features(self, input_df, features_list):
        '''Uses existing master encoding map to encode features in'''

        # Initialize df
        temp_df = input_df.copy()

        # Iterate through categorical columns and apply mapping
        for i in features_list:
            feature_map = self.master_encoding_map[i]
            temp_df[i] = temp_df[i].map(feature_map)

        # Return encoded df
        return temp_df
        

def get_group_dummies(input_df, feature_list, group_column):

    '''Gets the dummies for columns in feature_list, but groups
    them for all members of a group so result is binary (eg. if
    group member saw the same event twice, it will only show up
    as a binary 1 to keep table consistent).
    '''

    # Copy original df
    temp_df = input_df.copy()

    temp_df = pd.get_dummies(temp_df, columns=feature_list)

    # Group by an id column
    temp_agg = temp_df.groupby(group_column).sum()

    # Now change any values greater than 1 to 1
    temp_agg[temp_agg > 1] = 1

    return temp_agg
    


# TODO: add linear dt fxn
