import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from tabulate import tabulate

# Handle latex display
flatex = lambda s:r'\textbf{'+s+'}'

def split_data(data, target_col=''):
    """
    Split the data into matrix, target, and feature.

    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset.
    - target_col (str): The name of the column representing the target variable (optional).

    Returns:
    - matrix (ndarray): The matrix of features.
    - target (ndarray): The target variable.
    - feature (ndarray): The array of feature names.
    """
    data = data.copy()

    if len(target_col) != 0:
        target = data[target_col].astype(int).to_numpy()
        data.drop(columns=target_col, inplace=True)
    else:
        target = []

    matrix = data.to_numpy()
    feature = data.columns.to_numpy()

    return matrix, target, feature


def compute_vif(matrix, feature):
    """
    Compute the Variance Inflation Factor (VIF) for each feature to check for multicollinearity.

    Parameters:
    - matrix (ndarray): The matrix of features.
    - feature (ndarray): The array of feature names.

    Returns:
    - None
    """
    # Drop NaN values
    matrix = matrix[~np.isnan(matrix).any(axis=1)]

    print('CONTRÔLE DES COLINÉARITÉS\n\n')
    matrix = add_constant(matrix)
    vif = np.asarray([variance_inflation_factor(matrix, i) for i in range(matrix.shape[1])])[1:]
    print(tabulate(pd.DataFrame(vif, index=feature, columns=['vif']), headers='keys', tablefmt='grid'))

def build_preprocessor(matrix, target, plot_path):
    """
    Build preprocessor pipelines for data imputation and scaling.

    Parameters:
    - matrix (ndarray): The matrix of features.
    - target (ndarray): The target variable.

    Yields:
    - tuple: A tuple containing the name of the imputation method and its corresponding imputed matrix and target.
    """

    def drop_imputer(matrix, target):
        """
        Perform data imputation by dropping rows with missing values and scale the features.

        Parameters:
        - matrix (ndarray): The matrix of features.
        - target (ndarray): The target variable.

        Returns:
        - tuple: A tuple containing the imputed matrix and target.
        """
        # Drop missing values
        idx = np.argwhere(~np.isnan(matrix).any(axis=1)).ravel()
        imputed_matrix = matrix[idx]
        imputed_target = target[idx]

        scaler = StandardScaler()
        imputed_matrix = scaler.fit_transform(imputed_matrix)

        return imputed_matrix, imputed_target
    
    def iterative_imputer(matrix, target):
        """
        Perform data imputation using iterative imputer and scale the features.

        Parameters:
        - matrix (ndarray): The matrix of features.
        - target (ndarray): The target variable.

        Returns:
        - tuple: A tuple containing the imputed matrix and target.
        """
        estimator = LinearRegression()

        imputer = IterativeImputer(estimator=estimator, n_nearest_features=None, imputation_order='ascending', random_state=1)

        transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('iterative_imputer', imputer)
            ])

        imputed_matrix = transformer.fit_transform(matrix)

        # Analyse linear regression for col index = 3
        lr_model = transformer[1].imputation_sequence_[-1][2]
        lr_coef = lr_model.coef_.round(3)
        lr_intercept = lr_model.intercept_.round(3)

        y_lr = imputed_matrix[:,3]
        x_lr = np.delete(imputed_matrix,3,axis=1)

        lr_pred = lr_model.predict(x_lr)
        lr_residual = y_lr - lr_pred

        # Breush pagan
        bpagan_test = het_breuschpagan(lr_residual, add_constant(x_lr))
        test_result_text = f'Breusch Pagan test\n\nLagrange stat : {bpagan_test[0].round(3)} p-value : {bpagan_test[1]}\nF-statistic      : {bpagan_test[2].round(3)} p-value : {bpagan_test[3]}\nCoef             : {lr_coef}'
        # Plot for homeoskedasticity
        fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(15,10), layout='constrained')
        ax[0][0].scatter(lr_pred, lr_residual)
        ax[0][1].hist(lr_residual,orientation='horizontal',bins=25)

        ax[0][0].set_ylabel('residual')
        ax[0][0].set_xlabel('predicted')
        ax[0][1].set_xlabel('predicted')

        # Add text
        ax[1][0].set_axis_off()
        ax[1][0].set_ylim(0,1)
        ax[1][1].axes.remove()

        text = ax[1][0].text(0,1,test_result_text)

        fig.suptitle(flatex('Homeoskedasticity'), fontsize=16)
        fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
        fig.supxlabel('Figure 3.1: homeoskedasticity', x=0.5,y=0.25, ha='center', fontsize=8)

        fig.savefig(f'{plot_path}/fig3.1_homeoskedasticity.png')

        return imputed_matrix, target

    def knn_imputer(matrix, target):
        """
        Perform data imputation using KNN imputer and scale the features.

        Parameters:
        - matrix (ndarray): The matrix of features.
        - target (ndarray): The target variable.

        Returns:
        - tuple: A tuple containing the imputed matrix and target.
        """
        transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('knn_imputer', KNNImputer(n_neighbors=3,weights='distance'))
            ])

        imputed_matrix = transformer.fit_transform(matrix, target)

        return imputed_matrix, target

    # Perform different imputation methods
    imputed_matrix, imputed_target = drop_imputer(matrix, target)
    yield ('drop NaN', (imputed_matrix, imputed_target))

    imputed_matrix, imputed_target = iterative_imputer(matrix, target)
    yield ('linear regression', (imputed_matrix, imputed_target))
    
    imputed_matrix, imputed_target = knn_imputer(matrix, target)
    yield ('knn imputer', (imputed_matrix, imputed_target))

