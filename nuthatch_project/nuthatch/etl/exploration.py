import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

# Handle latex display
flatex = lambda s:r'\textbf{'+s+'}'

def describe_data(df, target):
    """
    Generate descriptive statistics for the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - target: str, optional
        The name of the target column for grouping analysis.

    Returns:
    None
    """
    # Group by target column and count occurrences
    group_df = df.groupby(target).count()
    print('RÉPARTITION DES VRAIS ET FAUX BILLETS\n\n')
    print(tabulate(group_df, headers='keys', tablefmt='grid'), '\n\n')
    
    describe_df = df.copy().describe()

    # Add NaN count and NaN ratio to the description DataFrame
    nan_count = df.isna().sum()
    nan_ratio = (nan_count / df.shape[0]) * 100
    describe_df.loc['nan_count'] = nan_count
    describe_df.loc['nan_ratio %'] = nan_ratio

    # Reorder rows for better readability
    describe_df = describe_df.loc[['count', 'nan_count', 'nan_ratio %', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    print('DESCRIPTION DU DATASET\n\n')
    print(tabulate(describe_df.round(2), headers='keys', tablefmt='grid'))
    
def plot_describe(data, target_col,plot_path):
    """
    Plot a scatter matrix and box plots of the features in the dataset with respect to the target variable.

    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset.
    - target_col (str): The name of the column representing the target variable.

    Returns:
    - None

    """
    def plot_scatter_matrix():
        # Create subplots for the scatter matrix
        fig, ax = plt.subplots(nrows=dim, ncols=dim, figsize=(12, 12))

        # Assign colors based on the target variable
        colors = np.asarray(['#d97e6a' if x == False else '#a0a882' for x in target])
    
        # Plot the scatter matrix
        scatter_matrix = pd.plotting.scatter_matrix(pd.DataFrame(x_data), ax=ax, diagonal='hist' , c=colors)
    
        # Hide grid lines for all subplots
        for i in range(dim):
            for j in range(dim):
                ax[i, j].grid(visible=False)
    
        # Set title, xlabel, and adjust layout
        fig.suptitle(flatex('Scatter Matrix'), fontsize=16)
        fig.supxlabel('Figure 1: Data Visualization', x=0.5, ha='center', fontsize=8)
        plt.subplots_adjust(bottom=0.15, wspace=0.05, hspace=0.05)

        fig.savefig(f'{plot_path}/fig1_scatter-matrix.png')

    def plot_box():
        fig, ax = plt.subplots(nrows=int(dim/2), ncols=int(dim/3), figsize=(12,10), layout='constrained')

        boxplot = data.boxplot(ax=ax, by=target_col, return_type='dict', patch_artist=True, showmeans=True, meanprops={'markerfacecolor':'black'})

        # Apply colors
        for key,val in boxplot.items():
            for i,color in enumerate(['C1','C2']):
                val['boxes'][i].set_facecolor(color)
                val['boxes'][i].set_alpha(0.5)

        # Change label
        for r in ax:
            for iax in r:
                iax.set_xlabel('')

        fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
        fig.suptitle(flatex('Box plots'), fontsize=16)
        fig.supxlabel('Figure 2: boxplot', x=0.5, ha='center', fontsize=8)
        fig.savefig(f'{plot_path}/fig2_boxplot.png')

    data = data.copy()  
    data.dropna(inplace=True)  

    x_data = data.drop(columns=target_col)
    target = data.loc[:, target_col]
    dim = x_data.shape[1]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_box()

    plot_scatter_matrix()
