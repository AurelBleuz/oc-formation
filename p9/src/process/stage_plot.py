import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sp_hierarchy
from sklearn.cluster import KMeans

translatex = lambda s: r'\textbf{'+s+'}'

def scree_plot(data):
    """
    Generate a scree plot to visualize the percentage of explained variance.

    Args:
    - scree (numpy.ndarray): Percentage of explained variance for each principal component.
    """
    xs = np.arange(data.shape[0]+1)
    scree_sum = data.cumsum()
    
    plt.ioff()
    fig,ax = plt.subplots(figsize=(8,8))

    # Bar plot
    bar_plot = ax.bar(
        x=xs[1:],
        height=data,
        color='C0'
    )
    # Line plot
    line_plot = ax.plot(
        xs[1:],
        scree_sum,
        marker='o',
        color='C1'
    )

    # Lines at 80%
    # Find x for m[x,80]
    arr = line_plot[0].get_xydata()
    arr = np.array([arr[arr[:,1] < 80][-1],arr[arr[:,1] > 80][0]])
    arr_left = np.array([arr[:,0],np.full(2,1)]).T
    arr_right = arr[:,1].T
    solv = np.linalg.solve(arr_left,arr_right)
    x_coord = (80-solv[1])/solv[0]
    # Draw
    ax.vlines(x=x_coord,ymin=0,ymax=80,linestyle='dashed',colors='black',alpha=0.25)
    ax.hlines(y=80,xmin=0,xmax=x_coord,linestyle='dashed',colors='black',alpha=0.25)
    
    # Axes configuration
    ax.set_xticks(ticks=xs[1:],labels=[f"C{x}" for x in xs[1:]])
    ax.set_xlabel(r'Composantes',fontsize=10)
    ax.set_ylabel(r'Inertie (\%)',fontsize=10)
    ax.set_title(r'\textbf{Scree plot}',fontsize=16,pad=25)
    return fig

def heatmap_plot(data,features):
    """
    Generate a heatmap to visualize the correlation matrix.

    Parameters:
    - data (numpy.ndarray): Correlation matrix.
    - features (list): List of feature names.
    """
    xs = np.arange(data.shape[0])
    ys = np.arange(data.shape[1])
    
    plt.ioff()
    
    fig = plt.figure(figsize=(15,6),layout='constrained')
    spec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[14,0.5])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])

    # Heatmap
    imap = ax0.imshow(data.T,vmin=-1,vmax=1,aspect='auto')
    # Text
    text = []
    for i in xs:
        for j in ys:
            # Set the color to white if data < 0
            if data.T[j,i] < 0:
                c = 'white'
            else:
                c = 'black'
            text.append(ax0.text(i,j,data.T[j,i].round(2),ha='center',va='center',fontsize=12,color=c))
    
    # Colorbar
    cbar = fig.colorbar(imap,cax=ax1)

    # Axes configuration
    ax0.grid(visible=False)
    ax1.grid(visible=False)
    ax0.set_xticks(xs,[f"C{x+1}" for x in xs])
    ax0.set_yticks(ys,features)
    ax0.set_title(r'\textbf{Composantes}',fontsize=16,pad=25)
    return fig

def correlation_plot(principal_components,explained_variance_ratio,features,ax_index):
    """
    Generate a correlation plot to visualize the circle of correlations.

    Args:
    - principal_components
    - features
    - explained_variance_ratio
    - ax_index
    """
    # Data
    ls = np.linspace(0,2*np.pi,100)
    xd = principal_components[ax_index[0]]
    yd = principal_components[ax_index[1]]
    fx = np.int8(explained_variance_ratio[ax_index[0]])
    fy = np.int8(explained_variance_ratio[ax_index[1]])

    # Plots
    plt.ioff()
    fig = plt.figure(figsize=(12,10),layout='constrained')
    spec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[10,2])
    ax = [fig.add_subplot(spec[0]),fig.add_subplot(spec[1])]
    # Draw circle & lines
    ax[0].plot(np.cos(ls),np.sin(ls))
    ax[0].vlines(x=0,ymin=-1,ymax=1,linestyle='dashed',colors='black',alpha=0.25)
    ax[0].hlines(y=0,xmin=-1,xmax=1,linestyle='dashed',colors='black',alpha=0.25)

    # Draw arrows
    texts = []
    for x,y,f in zip(xd,yd,features.keys()):
        ax[0].arrow(
            0,
            0,
            dx=x,
            dy=y,
            head_width=0.07,
            head_length=0.07,
            width=0.02,
            color='C1'
        )
        f = r'\textbf{'+f+r'}'
        texts.append(ax[0].text(x*1.25,y*1.25,f,ha='center'))

    # Legend
    ax[1].set_axis_off()
    s=""
    for k,v in features.items():
        s+=f"{k} : {v}\n"
    ax[1].text(
        x=0,
        y=1,
        s=s,
        verticalalignment='top',
        bbox={
            'boxstyle':'Round',
            'facecolor':(0.94,0.90,0.78,0.25)
        }
    )
    # Axes configuration
    ax[0].grid(visible=True)
    ax[0].set_xlabel(r'C'+str(ax_index[0]+1)+' ('+str(fx)+r'\%)',fontsize=10)
    ax[0].set_ylabel(r'C'+str(ax_index[1]+1)+' ('+str(fy)+r'\%)',fontsize=10)
    fig.suptitle(r'\textbf{Cercle des corrélations}',fontsize=16)
    return fig

def projection_plot(transformed_matrix,kmeans_mapped_index,dendrogram_mapped_index,samples,ax_indices):
    """
    Generate a scatter plot to visualize the projection of samples.

    Parameters:
    - *args: Variable-length argument list containing the data matrix, sample identifiers, and axis indices.
    """      
    # Plots
    plt.ioff()
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,8),layout='constrained')
    kmeans_centroid_coordinate = {}
    dendrogram_centroid_coordinate = {}
    for iax,cluster in zip(enumerate(ax),[kmeans_mapped_index,dendrogram_mapped_index]):
        key_list = list(cluster.keys())
        key_list.sort()
        color_list = [f"C{x}" for x in range(1,len(key_list)+1)]
        # Scatter plot
        for k,c in zip(key_list,color_list):
            v = cluster[k]
            x_value = transformed_matrix[[v],ax_indices[0]]
            y_value = transformed_matrix[[v],ax_indices[1]]
            iax[1].scatter(
                x=x_value,
                y=y_value,
                c=c,
                label=f"cluster {k}"
            )
            # Plot centroids
            x_centroid = x_value.mean()
            y_centroid = y_value.mean()
            if iax[0] == 0:
                kmeans_centroid_coordinate[f"cluster_{k}"] = [x_centroid,y_centroid]
            else:
                dendrogram_centroid_coordinate[f"cluster_{k}"] = [x_centroid,y_centroid]
                
            iax[1].scatter(
                x=x_centroid,
                y=y_centroid,
                c=c,
                marker='^',
                s=70,
                label='centroid'
            )
            # Plot link between centroids and samples
            for x_coord,y_coord in zip(x_value[0],y_value[0]):
                iax[1].plot(
                    [x_centroid,x_coord],
                    [y_centroid,y_coord],
                    c=c,
                    linewidth=0.5,
                    alpha=0.33
                )
            
        # Vertical / horizontal lines
        iax[1].vlines(x=0,ymin=transformed_matrix[:,ax_indices[1]].min(),ymax=transformed_matrix[:,ax_indices[1]].max(),linestyle='dashed',colors='black',alpha=0.25)
        iax[1].hlines(y=0,xmin=transformed_matrix[:,ax_indices[0]].min(),xmax=transformed_matrix[:,ax_indices[0]].max(),linestyle='dashed',colors='black',alpha=0.25)
        # Ax configuration
        ymax = iax[1].get_ylim()[1]*1.25
        iax[1].set_ylim(ymax=ymax)
        iax[1].set_xlabel(r'C'+str(ax_indices[0]+1),fontsize=10)
        iax[1].set_ylabel(r'C'+str(ax_indices[1]+1),fontsize=10)
        iax[1].grid(visible=True)
        
        if iax[0] == 0:
            iax[1].set_title('K-means clustering',fontsize=14)
        else:
            iax[1].set_title('Dendrogram clustering',fontsize=14)
            
    ax[0].legend(loc=9,ncols=5,fontsize=10,markerscale=0.5)
    ax[1].legend(loc=9,ncols=5,fontsize=10,markerscale=0.5)
    fig.suptitle(r'\textbf{Projection des échantillons}',fontsize=16)
    
    return fig

def dendrogram_plot(matrix,feature,n_cluster=3,tresh=30):
    """
    Plot a dendrogram based on hierarchical clustering of the input matrix.

    Args:
    - matrix (array-like): The input matrix for hierarchical clustering.
    - t (float): The threshold for cutting the dendrogram.

    Returns:
    dict: A dictionary containing the following information:
        - 'mapped_index': Mapping of clusters to matrix indices.
        - 'dendrogram_data': Data from the dendrogram plot.
        - 'linkage_matrix': The hierarchical clustering linkage matrix.
    """
    linkage_matrix = sp_hierarchy.linkage(matrix, method='ward')
    # Cluster matrix -1 to align with kmeans clustering
    cluster_matrix = sp_hierarchy.fcluster(linkage_matrix, criterion='maxclust',t=n_cluster) - 1

    # Mapper les clusters
    cluster_map = {f"{k}":[] for k in range(n_cluster)}
    for cluster,index in zip(cluster_matrix,range(len(matrix))):
        cluster_map[f"{cluster}"].append(index)

    # Plotting dendrogram
    plt.ioff()
    dendrogram_figure = plt.figure(figsize=(12,10),layout='constrained')
    centroid_figure = plt.figure(figsize=(15,6),layout='constrained')
    ax0 = dendrogram_figure.subplots()
    ax1,ax2 = centroid_figure.subplots(nrows=1,ncols=2)
    
    # ratio due to the coordiante transformation of the ellipse
    fig_ratio = dendrogram_figure.get_figheight() / dendrogram_figure.get_figwidth()
        
    dendrogram_data = sp_hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=tresh,
        distance_sort=True,
        ax=ax0
    )
    
    ax0.axhline(y=tresh, color='black', linestyle='dashed', alpha=0.25)

    # Create legend for the clusters
    # Draw a FancyBox
    x,y = (0.85,0.84)
    fbox = matplotlib.patches.FancyBboxPatch(
        xy=(x,y),
        width=0.12,
        height=0.12,
        boxstyle='Round, pad=0.01',
        facecolor=(0.94,0.90,0.78,0.25),
        alpha=0.75,
        transform=ax0.transAxes
    )
    ax0.add_patch(fbox)

    color_list = []
    for c in dendrogram_data['color_list']:
        if c not in color_list:
            color_list.append(c)
    color_list = color_list[:-1]
    
    x,y = (0.85,0.95)
    for i,c in enumerate(color_list):
        # Add a colored circle
        circle_patch = matplotlib.patches.Ellipse(
            xy=(x,y),
            width=0.005,
            height=0.005/fig_ratio,
            color=c,
            transform=ax0.transAxes
        )
        ax0.add_patch(circle_patch)
        
        # Add text
        ax0.text(
            x=x+0.01,
            y=y,
            s=f"cluster {i}",
            verticalalignment='center',
            transform=ax0.transAxes
        )
        
        # Update coordinates
        x,y = (x,y-0.025)
        
    # Centroids plot
    xs = np.arange(len(feature))
    for c,(k,v) in enumerate(cluster_map.items()):
        # plot cluster 1 & 2 together
        ydata = matrix[v].mean(axis=0)
        if k in ['0','1']:
            ax1.plot(
                xs,
                ydata,
                color=f"C{c+1}",
                label=f"cluster {k}"
            )
        else:
            ax2.plot(
                xs,
                ydata,
                color=f"C{c+1}",
                label=f"cluster {k}"
            )
         
    # Plot configuration
    ax0.set_title(r"\textbf{Dendrogramme}", fontsize=16, pad=25)
    ax0.set_xlabel(r"\'Echantillons", fontsize=10)
    ax0.set_ylabel("Distance", fontsize=10)
    for iax,title in zip([ax1,ax2],['clusters 0 et 1','clusters 2,3,4']):
        iax.set_xticks(ticks=xs,labels=feature,rotation='vertical')
        iax.set_title('Graph centroids '+title,fontsize=12,pad=10)
        iax.legend()
    centroid_figure.suptitle(r'\textbf{Visualisation parallèle des centroides}',fontsize=16)
    return dendrogram_figure,centroid_figure,cluster_map

def elbow_plot(matrix,t=2):
    """
    Plot an elbow chart to determine the optimal number of clusters.

    Parameters:
    - inertia_list (list): List of inertia values for different numbers of clusters.

    Returns:
    None
    """
    def compute_inertia(scaled_matrix):
        """
        Compute the inertia (within-cluster sum of squares) for different numbers of clusters.
    
        Parameters:
        - matrix (array-like): The input matrix for k-means clustering.
    
        Returns:
        list: List of inertia values for different numbers of clusters.
        """
        # Compute inertia for the elbow method
        inertia_list = []
        for i in range(1, 11):
            # Initialize k-means
            kmeans = KMeans(n_clusters=i,init='k-means++',n_init=100)
            kmeans.fit(scaled_matrix)
            inertia_list.append(kmeans.inertia_)
        return inertia_list
        
    # Data
    inertia_list = compute_inertia(matrix)
    xs = np.arange(1, len(inertia_list) + 1)

    # Plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 6))

    elbow = ax.plot(xs, inertia_list)
    elbow_xy = elbow[0].get_xydata()[t, :]
    ax.vlines(x=elbow_xy[0], ymin=0, ymax=elbow_xy[1], linestyle='dashed', colors='black', alpha=0.25)
    ax.hlines(y=elbow_xy[1], xmin=1, xmax=elbow_xy[0], linestyle='dashed', colors='black', alpha=0.25)

    # Axis configuration
    ax.set_xticks(xs)
    ax.set_title(r'\textbf{K-means: Elbow}', fontsize=16, pad=25)
    ax.set_xlabel('Nombre de clusters', fontsize=10)
    ax.set_ylabel('Inertie', fontsize=10)

    return fig

def box_plot(matrix, feature, cluster_map, n_cluster=2,**kwarg):
    """
    Generate a grouped box plot for specified features across different clusters.

    Parameters:
    - matrix (numpy.ndarray): Input data matrix.
    - feature (list): List of feature names to be plotted.
    - cluster_map (dict): Mapping of cluster labels to corresponding data indices.
    - n_cluster (int, optional): Number of clusters to include in the plot. Default is 2.
    **kwarg : figure configuration, figsize, layout
    
    Returns:
    - figure
    """
    # Data preparation
    xs = range(1, n_cluster + 1)
    cluster_map = cluster_map[:n_cluster]

    # Plot
    plt.ioff
    fig, ax = plt.subplots(nrows=4, ncols=3, **kwarg)
    f = 0
    color_list = ['C1', 'C2']
    
    for row in range(len(ax)):
        for col in range(len(ax[0])):
            if f < len(feature):
                cluster_data = [matrix[cluster][:, f] for cluster in cluster_map]
                # compute the means
                mean_data = [x.mean() for x in cluster_data]
                # find the std
                data_std = [x.std() for x in cluster_data]
                # draw the plot
                bplot = ax[row][col].boxplot(cluster_data, patch_artist=True, showmeans=True,meanprops={'markerfacecolor':'black'},showfliers=False)
                ax[row][col].set_title(f"{feature[f]}", pad=16)
                ax[row][col].set_xticks(xs, [f'cluster {i}' for i in range(n_cluster)])

                for patch, color in zip(bplot['boxes'], color_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
                
                # display the mean value
                ylimit = ax[row][col].get_ylim()
                for i,m in enumerate(mean_data):
                    symbol_b = r'$\mu$'
                    ax[row][col].text(x=i+1,y=ylimit[1],s=f"{symbol_b} {round(m,3)}",horizontalalignment='center',fontsize=10)
                
                ax[row][col].set_ylim(ymin=0,ymax=ylimit[1]*1.15)

                f += 1

    ax[-1][-1].remove()
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
    fig.suptitle(r'\textbf{Détail des clusters 0 et 1}')

    return fig

def plot_worldmap(geomap_file):
    """
    Plot world map with clusters highlighted.

    Parameters:
    - geomap_file (str): Path to the GeoJSON file containing geographical data.

    Returns:
    None
    """
    # Import GeoJson dataset
    gdf_worldmap = gpd.read_file(geomap_file)

    # Define color list for clusters
    color_list = ['C1', 'C2', 'C3', 'C4', 'C5', '#bacbcf7e']
    
    # Map cluster indices to colors and create a new 'color' column
    gdf_worldmap['color'] = gdf_worldmap['cluster'].map(lambda i: color_list[i])

    # Create a subplot for the world map
    plt.ioff()
    fig, ax = plt.subplots(figsize=(18,16),layout='constrained')
    
    # Plot the world map with cluster colors
    gdf_worldmap.plot(color=gdf_worldmap['color'], alpha=0.7, edgecolor='#373c3dcc', linewidth=0.2, ax=ax)

    # Legend
    x, y = (0.05, 0.5)
    offset = 0
    fig_ratio = fig.get_figheight() / fig.get_figwidth()

    # Iterate over color list to create legend patches and text
    for i, c in enumerate(color_list[:-1]):
        # Create a circle patch for each cluster
        circle_patch = matplotlib.patches.Ellipse(
            xy=(x, y - offset),
            width=0.005,
            height=0.005 / fig_ratio,
            color=c,
            transform=ax.transAxes
        )
    
        # Add the circle patch to the subplot
        ax.add_patch(circle_patch)

        # Add legend text for each cluster
        legend_text = ax.text(
            x=x + 0.01,
            y=y - offset,
            s=f"cluster {i}",
            verticalalignment='center',
            fontsize=10,
            transform=ax.transAxes
        )

        # Update offset for the next legend entry
        offset += 4 * circle_patch.get_height()

    # Set subplot title
    ax.set_title(r'\textbf{Projection géographique des clusters}', fontsize=16,pad=25)

    # Remove ticks and grid lines
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(visible=False)

    return fig
