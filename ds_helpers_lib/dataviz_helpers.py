import  numpy  as  np
import  matplotlib.pyplot as plt
import seaborn as sns

def plot_pairs(input_df, target_col, grid_col_num):
    '''Use Seaborn to plot the relationships between feat cols against one
    target column. Define number of columns in plot grid with grid_col_num'''
    
    # Define dimensions of grid
    feat_cols_only = input_df.drop(target_col, axis=1).columns
    n_cols = grid_col_num
    n_rows = len(feat_cols_only)//n_cols
    print(len(feat_cols_only), n_rows)
    
    # Split feats list into sublists to match grid dimensions
    splits = np.array_split(feat_cols_only, n_rows)
    splits = splits[::-1]
    
    # Plot pair based on these target and sublists of features
    for i in splits:
        sns.pairplot(input_df, y_vars=target_col, x_vars=i)
    


def scatter_boxplot(my_df, xlabels, ylabels, title, fig_size= [10,6]):
    '''Style formatting for boxplots'''

    plt.subplots(1,1, figsize=fig_size, facecolor='white')
    plt.style.use('fivethirtyeight')
    sns.boxplot(data=my_df, saturation= 0.2)
    sns.stripplot(data=my_df, jitter=True, size= 8.5, linewidth= 1, edgecolor= 'black')
    plt.xlabel(xlabels, fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontstyle='italic')
    plt.ylabel(ylabels, fontweight='bold', fontsize=20)
    plt.yticks(fontsize=16, fontstyle='italic')
    plt.title(title, fontweight= 'bold', fontstyle='italic', fontsize=24)
    plt.grid(color='white')