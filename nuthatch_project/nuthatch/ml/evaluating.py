import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, rand_score
from sklearn.decomposition import PCA
from scipy.special import expit
from tabulate import tabulate

# Handle latex display
flatex = lambda s:r'\textbf{'+s+'}'

def compare_model(matrix, target, feature, model_path, plot_path):
    """
    Compare the performance of logistic regression and KMeans clustering models.

    Parameters:
    - matrix (ndarray): The matrix of features.
    - target (ndarray): The target variable.
    - feature (ndarray): The features.

    Returns:
    - None
    """
    def compute_pca(matrix):
        """
        Perform Principal Component Analysis (PCA) on the input matrix.

        Parameters:
        - matrix (ndarray): The matrix of features.

        Returns:
        - ndarray: The transformed matrix after PCA.
        """
        pca = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=3)),
                ('pca', PCA(n_components=2))
            ])

        pca_matrix = pca.fit_transform(matrix)

        return pca_matrix

    def remap_kmeans(kmeans_pred, expit_proba):
        """
        Remap the labels predicted by KMeans clustering based on the probabilities from logistic regression.

        Parameters:
        - kmeans_pred (ndarray): The predicted labels by KMeans.
        - expit_proba (ndarray): The predicted probabilities from logistic regression.

        Returns:
        - ndarray: The remapped labels.
        """
        # Test if probability of class 0 is higher than class 1, then remap
        class_0_proba = expit_proba[kmeans_y_pred == 0].mean(axis=0)
        idx_class_0 = np.argmax(class_0_proba)
        idx_class_0 = kmeans_y_pred == idx_class_0
        
        kmeans_y_pred[idx_class_0] = 0
        kmeans_y_pred[~idx_class_0] = 1

        return kmeans_y_pred

    def report():
        """
        Print evaluation reports for logistic regression and KMeans models.
        """
        # Logistic regression evaluation
        target_label = ['Faux billets', 'Vrais billets']
        print(f'\nLOGISTIC REGRESSION EVALUATION\n{classification_report(target, expit_y_pred, target_names=target_label)}')

        # KMeans evaluation
        score = {'truth_compare': rand_score(target, kmeans_y_pred), 'expit_compare': rand_score(expit_y_pred, kmeans_y_pred)}
        df = pd.DataFrame.from_dict(score, orient='index', columns=['rand_score'])
        print(f'\nKMEANS EVALUATION WITH GROUND TRUTH\n{classification_report(target, kmeans_y_pred, target_names=target_label)}')
        print(f'\nKMEANS EVALUATION WITHOUT GROUND TRUTH\n{tabulate(df, headers="keys", tablefmt="grid")}')

    def display_confusion_matrix():
        """
        Print confusion matrices for logistic regression and KMeans models.
        """
        # Confusion matrices
        expit_cmatrix = confusion_matrix(target, expit_y_pred)
        kmeans_cmatrix = confusion_matrix(target, kmeans_y_pred)

        df_expit = pd.DataFrame(expit_cmatrix, index=['Actual 0', 'Actual 1'], columns=['Predict 0', 'Predict 1'])
        df_kmeans = pd.DataFrame(kmeans_cmatrix, index=['Actual 0', 'Actual 1'], columns=['Predict 0', 'Predict 1'])

        print(f'\n\nLOGISTIC REGRESSION CONFUSION MATRIX\n{tabulate(df_expit, headers="keys", tablefmt="grid")}')
        print(f'\n\nKMEANS CONFUSION MATRIX\n{tabulate(df_kmeans, headers="keys", tablefmt="grid")}')

    def plot_cluster():
        """
        Plot clusters from PCA for visual comparison.
        """
        fig, ax = plt.subplots(ncols=3, figsize=(15,10), layout='constrained')
        for i, (name, color) in enumerate([('True', target), ('Expit', expit_y_pred), ('KMeans', kmeans_y_pred)]):
            color = [f'C{s}' for s in color]
            ax[i].scatter(pca_matrix[:, 0], pca_matrix[:, 1], c=color)
            ax[i].set_title(name, fontsize=10)
        fig.suptitle(flatex('Model Comparison'), fontsize=16)
        fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
        fig.supxlabel('Figure 4: models comparison', x=0.5, ha='center', fontsize=8)
        fig.savefig(f'{plot_path}/fig4_models-comparison.png')

    # Load models
    with open(f'{model_path}/expit_model.pkl', 'rb') as f:
        expit_model = pickle.load(f)
    with open(f'{model_path}/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    # Logistic regression predictions
    expit_y_pred = expit_model.predict(matrix)
    expit_proba = expit_model.predict_proba(matrix)
    
    # KMeans clustering predictions
    kmeans_y_pred = kmeans_model.predict(matrix)
    kmeans_y_pred = remap_kmeans(kmeans_y_pred, expit_proba)

    # PCA for visual comparison
    pca_matrix = compute_pca(matrix)
    
    report()
    display_confusion_matrix()
    plot_cluster()

    # Extract KBest features
    # kbest_mask = expit_model[2].get_support()
    # print(feature[kbest_mask])

    # Sigmoid coef & bias
    #coef = expit_model[3].coef_
    #intercept = expit_model[3].intercept_
    #data = StandardScaler().fit_transform(matrix)

    #sum_z = (data[:, 3:] * coef).sum(axis=1) + intercept 
    #fsigmoid = expit(sum_z)
    # print(fsigmoid[0])
    # print(expit_proba[0])

def use_model(data_file,id_col,model_path):
    """
    """
    col = ['diagonal','height_left','height_right','margin_low','margin_up','length','id']
    dataset = pd.read_csv(data_file,usecols=col,sep=',')

    matrix = dataset.drop(columns=[id_col]).to_numpy()
    ident = dataset[id_col]
    feature = dataset.drop(columns=[id_col]).columns.to_numpy()
    
    # Load models
    with open(f'{model_path}/expit_model.pkl', 'rb') as f:
        expit_model = pickle.load(f)
    with open(f'{model_path}/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
  
    # Use probability
    expit_proba = expit_model.predict_proba(matrix)
    kmeans_predict = kmeans_model.predict(matrix)

    result_proba = {'proba_faux':(expit_proba[:,0] * 100).round(2),'proba_vrai':(expit_proba[:,1] * 100).round(2)}
    result_kmeans = {'cluster':kmeans_predict}
    
    cluster_feature = feature[kmeans_model[2].get_support()]
    cluster_center = {f'cluster {idx}':v for idx,v in zip(range(2),kmeans_model[3].cluster_centers_)}

    print(
        f'LOGISTIQUE RÉGRESSION\n' \
        f'{tabulate(pd.DataFrame.from_dict(result_proba,orient="index",columns=ident).T,headers="keys",tablefmt='grid')}\n\n')
    print(
        f'KMEANS\n' \
        f'{tabulate(pd.DataFrame.from_dict(result_kmeans,orient='index',columns=ident).T,headers='keys',tablefmt='grid')}\n\n'
    )
    
    print(
        f'ANALYSE DES CENTROÏDS\n' \
        f'{tabulate(pd.DataFrame.from_dict(cluster_center,orient="index",columns=cluster_feature),headers="keys",tablefmt="grid")}'
    )
