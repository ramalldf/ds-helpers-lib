import  numpy  as  np
import  matplotlib.pyplot as plt
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