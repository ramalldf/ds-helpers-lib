# Import seaborn
import seaborn as sns

def plot_pairs(df, target_col, one_vs_all=False):
    '''Use Seaborn to plot the relationships between feat cols against one
    another. Can use target column to separate classes by hue'''
    
    if one_vs_all is False:
        # Use pairplot and set the hue to be our class
        sns.pairplot(df, hue=target_col) 

        # Show the plot
        plt.show()
    
    else:
        sns.pairplot(df, y_vars=target_col, x_vars=df.drop(target_col, axis=1).columns)
        plt.show()


