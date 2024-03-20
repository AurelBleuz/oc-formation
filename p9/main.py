import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import src.process.stage_prepare as sprep
import src.process.stage_pca as spca
import src.process.stage_plot as splot

def load_configuration(config_path):
    """
    Load the configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - dict: Loaded configuration.
    """
    with open(config_path, 'r') as fp:
        return json.load(fp)

def load_and_clean_data(config):
    """
    Load and clean data based on the provided configuration.

    Parameters:
    - config (dict): Configuration containing parameters for data cleaning.

    Returns:
    - pd.DataFrame: DataFrame containing cleaned data.
    """
    food_availability_data = sprep.clean_food_availability_data(**config['food_availability_data'])
    population_data = sprep.clean_population_data(**config['population_data'])
    gdp_data = sprep.clean_gdp_data(**config['gdp_data'])

    merged_df = pd.merge(food_availability_data, population_data, on='Zone').fillna(0)
    merged_df = pd.merge(merged_df, gdp_data, on='Zone')

    merged_df['PIB par habitant'] = (merged_df['PIB'] / merged_df['Population']).round()
    
    return merged_df

def process_data(dataset):
    """
    Perform data processing, including clustering, PCA, and creating plots.

    Parameters:
    - dataset (dict): Dictionary containing data.

    Returns:
    - Tuple[matplotlib.figure.Figure]: Figures generated during processing.
    """
    dendrogram_figure, centroid_figure, dendrogram_cluster = splot.dendrogram_plot(
        matrix=dataset['scaled_matrix'], feature=list(dataset['features'].values()), n_cluster=5, tresh=15
    )
    dataset.update({'dendrogram_cluster': dendrogram_cluster})

    boxplot_figure = splot.box_plot(
        dataset['matrix'],
        list(dataset['features'].values()),
        list(dataset['dendrogram_cluster'].values()),
        figsize=(12, 20),
        layout='constrained'
    )

    elbow_figure = splot.elbow_plot(matrix=dataset['scaled_matrix'], t=4)

    kmeans_obj, kmeans_cluster = spca.perform_kmeans(dataset['scaled_matrix'], 5)
    kmeans_cluster = spca.cluster_transmap(kmeans_cluster, dendrogram_cluster)
    dataset.update({'kmeans_cluster': kmeans_cluster, 'kmeans_obj': kmeans_obj})

    pca_result = spca.perform_pca(dataset['scaled_matrix'], 4)
    dataset.update(pca_result)

    scree_figure = splot.scree_plot(pca_result['explained_variance_ratio'])
    heatmap_figure = splot.heatmap_plot(pca_result['principal_components'], dataset['features'].keys())

    ax_index = [(0, 1), (1, 2)]
    for i in enumerate(ax_index):
        correlation_figure = splot.correlation_plot(
            pca_result['principal_components'],
            pca_result['explained_variance_ratio'],
            dataset['features'],
            i[1]
        )
        correlation_figure.savefig(f"./plot/correlation_{i[0]}.png")
        projection_figure = splot.projection_plot(
            pca_result['transformed_matrix'],
            kmeans_cluster,
            dendrogram_cluster,
            dataset['samples'].keys(),
            i[1]
        )
        projection_figure.savefig(f"./plot/projection_{i[0]}.png")

    # geomap_figure = splot.plot_worldmap('./file/geodata.geojson')

    return dataset,(dendrogram_figure, centroid_figure, boxplot_figure, elbow_figure, scree_figure, heatmap_figure) # geomap_figure

def save_outputs(dataset, figures):
    """
    Save the dataset and generated figures.

    Parameters:
    - dataset (dict): Dictionary containing the dataset.
    - figures (Tuple[matplotlib.figure.Figure]): Figures generated during processing.
    """
    with open('./pkl/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f, protocol=-1)

    for name, fig in zip(['dendrogram', 'centroid', 'boxplot', 'elbow', 'scree', 'heatmap'],
                         figures):
        fig.savefig(f'./plot/{name}.png')

    # figures[-1].savefig('./plot/geomap_dendrogram.png', bbox_inches='tight')

if __name__ == "__main__":
    # Load the configuration
    config = load_configuration('./conf/file.json')
    matplotlib.style.use(config['mplstyler']['path'])

    # Load and clean data
    merged_df = load_and_clean_data(config)
    dataset = spca.split_data(merged_df)

    # Data processing and figure generation
    dataset,figures = process_data(dataset)

    # Save outputs
    save_outputs(dataset, figures)
