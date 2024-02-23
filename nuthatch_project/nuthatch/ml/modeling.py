import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from tabulate import tabulate

# Handle latex display
flatex = lambda s:r'\textbf{'+s+'}'

def build_model(x, y, model_path):
    """
    Build and train machine learning models.

    Parameters:
    - x (ndarray): The matrix of features.
    - y (ndarray): The target variable.

    """
    # Imputer
    imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

    # Define supervised and unsupervised models
    sup_model = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('imputer', imputer),
            ('feat_select', SelectKBest(k=3)),
            ('estimator', LogisticRegression(max_iter=500))
        ])

    usup_model = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('imputer', imputer),
            ('feat_select', SelectKBest(k=3)),
            ('estimator', KMeans(n_clusters=2))
        ])
    # Train models
    sup_model.fit(x, y)
    usup_model.fit(x, y)

    # Save models
    model = {'expit': sup_model, 'kmeans': usup_model}
    for k, v in model.items():
        with open(f'{model_path}/{k}_model.pkl', 'wb') as f:
            pickle.dump(v, f, -1)

def evaluate_model(matrix_target, plot_path, model='expit', n_split=3):
    """
    Evaluate the model using cross-validation and display performance metrics.

    Parameters:
    - matrix_target (list): List of tuples containing matrices and corresponding targets.
    - model (str): The model to evaluate ('expit' for logistic regression or 'kmeans' for KMeans).
    - n_split (int): Number of splits for cross-validation.

    Returns:
    - None
    """
    def eval_supmodel(matrix_target, skf):
        """
        Evaluate supervised model performance.

        Parameters:
        - matrix_target (list): List of tuples containing matrices and corresponding targets.
        - skf (StratifiedKFold): Stratified K-Folds cross-validator.

        Returns:
        - None
        """
        estimator = Pipeline(
            steps=[
                ('KBest', SelectKBest(k=3)),
                ('expit', LogisticRegression(max_iter=500))
            ])

        recall_0 = make_scorer(recall_score, pos_label=0)
        result = []
        for i in matrix_target:
            method = i[0]

            # Keep ridge imputer's values
            if method == 'linear regression':
                br_matrix, br_target = i[1]

            matrix, target = i[1]

            cv_result = cross_validate(
                estimator,
                matrix,
                target,
                scoring={
                    'log_loss_score': 'neg_log_loss',
                    'roc_auc_score': 'roc_auc'
                },
                cv=skf)

            df_result = pd.DataFrame.from_dict(cv_result, orient='columns').describe()
            result.append((method, df_result.loc[['mean', 'min', 'max']]))

        print(f'CROSS VALIDATION : KFOLD\nSplits {":":>6} {n_split}\n\n')

        for r in result:
            print(f'Methode d\'imputation : {r[0]}\n{r[1]}\n')

        # Plot Learning Curve
        fig, ax = plt.subplots(figsize=(15,10), layout='constrained')
        lcurve = LearningCurveDisplay.from_estimator(
            estimator=estimator,
            X=br_matrix,
            y=br_target,
            train_sizes=np.linspace(0.25, 1.0, 20),
            scoring='neg_log_loss',
            cv=skf,
            shuffle=True,
            random_state=1,
            ax=ax,
            fill_between_kw={'alpha': 0.15})

        ax.set_title(flatex('Learning Curve'), fontsize=16)
        fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
        fig.supxlabel('Figure 3: learning curve', x=0.5, ha='center', fontsize=8)
        fig.savefig(f'{plot_path}/fig3_learning-curve.png')

    def eval_usupmodel(matrix_target, skf):
        """
        Evaluate unsupervised model performance.

        Parameters:
        - matrix_target (list): List of tuples containing matrices and corresponding targets.
        - skf (StratifiedKFold): Stratified K-Folds cross-validator.

        Returns:
        - None
        """
        estimator = Pipeline(
            steps=[
                ('KBest', SelectKBest(k=3)),
                ('kmeans', KMeans(n_clusters=2))
            ])

        result = []
        for i in matrix_target:
            method = i[0]
            matrix, target = i[1]
            cv_result = cross_validate(estimator, matrix, target,
                scoring={'v_measure': 'v_measure_score','rand_index': 'rand_score','adj_rand_index': 'adjusted_rand_score'},
                cv=skf)
            df_result = pd.DataFrame.from_dict(cv_result, orient='columns').describe()
            result.append((method, df_result.loc[['mean', 'min', 'max']]))

        print(f'CROSS VALIDATION : KFOLD\nSplits {":":>6} {n_split}\n\n')

        for r in result:
            print(f'Methode d\'imputation : {r[0]}\n{r[1]}\n')

    # Cross validate and score
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)

    if model == 'expit':
        eval_supmodel(matrix_target, skf)
    elif model == 'kmeans':
        eval_usupmodel(matrix_target, skf)
