import os
import sys
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.style
from sklearn.model_selection import train_test_split

import nuthatch.etl.exploration as nut_exploration
import nuthatch.ml.preprocessing as nut_preprocessing
import nuthatch.ml.modeling as nut_modeling
import nuthatch.ml.evaluating as nut_evaluating
from nuthatch import MPL_STYLER, TRAIN_FILE, PLOT_PATH

# Matplotlib styler
matplotlib.style.use(MPL_STYLER)
# Handle latex display
flatex = lambda s:r'\textbf{'+s+'}'

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def main():
    """Main function of the program."""
    clear_screen()
    
    # Display help message if no arguments provided or '--help' option used
    if (len(sys.argv) <= 1) or (sys.argv[1] == '--help'):
        clear_screen()
        message_help = 'Détecteur de faux billets :\n' \
            f'{"--describe-dataset":<30} : affiche les informations du dataset d\'entrainement\n' \
            f'{"--describe-model nom_du_modèle":<30} : affiche l\'évaluation des différentes options de preprocessing\n' \
            f'{"--build-model path [option]":<30} : refit le model et l\'export dans <path>\n' \
            f'{"":<4}{"option --eval":<26} : afficher l\'évaluation\n' \
            f'{"--use-model path dataset":<30} : utiliser et comparer les modèles sur un dataset d\'évaluation \n' \
            f'{"--help":<30} : affiche ce message \n' \

        return print(message_help)

    # Read dataset and preprocess
    dataset = pd.read_csv(TRAIN_FILE, usecols=['diagonal','height_left','height_right','margin_low','margin_up','length','is_genuine'], sep=';') 
    matrix, target, feature = nut_preprocessing.split_data(dataset,'is_genuine')

    if sys.argv[1] == '--describe-dataset':
        clear_screen()

        # Describe dataset and perform data exploration
        nut_exploration.describe_data(dataset,'is_genuine')
        input('\n...\n')
        nut_preprocessing.compute_vif(matrix, feature)
        nut_exploration.plot_describe(dataset,'is_genuine', plot_path=PLOT_PATH)
    
    elif sys.argv[1] == '--describe-model':
        clear_screen()

        if (len(sys.argv) < 3) or sys.argv[2] not in ['expit','kmeans']:
            raise ValueError('Utilisation : --describe-model [nom_du_model : expit / kmeans]')

        # Build preprocessor and evaluate model
        gen_preprocessor = nut_preprocessing.build_preprocessor(matrix, target, plot_path=PLOT_PATH)
        nut_modeling.evaluate_model(gen_preprocessor, model=sys.argv[2], plot_path=PLOT_PATH)

    elif sys.argv[1] == '--build-model':
        clear_screen()

        if (len(sys.argv) < 3) or not os.path.isdir(sys.argv[2]):
            raise FileNotFoundError('Vous devez fournir un dossier d\'exportation')

        model_path = sys.argv[2]

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(matrix, target, stratify=target)
        nut_modeling.build_model(x_train, y_train, model_path)

        # Optionally evaluate model if --eval option provided
        if (len(sys.argv) == 4) and (sys.argv[3] == '--eval'):
            nut_evaluating.compare_model(x_test, y_test, feature, model_path, plot_path=PLOT_PATH)

    elif sys.argv[1] == '--use-model':
        clear_screen()

        if (len(sys.argv) < 4) or not os.path.isdir(sys.argv[2]):
            raise FileNotFoundError('Chemin incorrect ou dataset absent')

        model_path = sys.argv[2]

        # Build models if not already present
        for file in ['expit','kmeans']:
            if not os.path.exists(f'{model_path}/{file}_model.pkl'):
                x_train, x_test, y_train, y_test = train_test_split(matrix, target, stratify=target)
                nut_modeling.build_model(x_train, y_train, model_path)
                break

        if not os.path.exists(sys.argv[3]):
            raise FileNotFoundError('Fichier introuvable')

        data_file = sys.argv[3]
        nut_evaluating.use_model(data_file, 'id', model_path)

    else:
        raise ValueError('Paramètres incorrects < --help > pour afficher l\'aide')

if __name__ == '__main__':
     if len(sys.argv) <= 1:
         sys.exit('Trop peu d\'arguments pour exécuter le script')
     main()

# def main():
#     """
#     """
#     clear_screen()
#     if (len(sys.argv) <= 1) or (sys.argv[1] == '--help'):
#         clear_screen()
#         message_help = 'Détecteur de faux billets :\n' \
#             f'{"--describe-dataset":<30} : affiche les informations du dataset d\'entrainement\n' \
#             f'{"--describe-model nom_du_modèle":<30} : affiche l\'évaluation des différentes options de preprocessing\n' \
#             f'{"--build-model path [option]":<30} : refit le model et l\'export dans <path>\n' \
#             f'{"":<4}{"option --eval":<26} : afficher l\'évaluation\n' \
#             f'{"--use-model path dataset":<30} : utiliser et comparer les modèles sur un dataset d\'évaluation \n' \
#             f'{"--help":<30} : affiche ce message \n' \
# 
#         return print(message_help)
# 
#     dataset = pd.read_csv(TRAIN_FILE,usecols=['diagonal','height_left','height_right','margin_low','margin_up','length','is_genuine'],sep=';') 
#     matrix, target, feature = nut_preprocessing.split_data(dataset,'is_genuine')
# 
#     if sys.argv[1] == '--describe-dataset':
#         clear_screen()
# 
#         nut_exploration.describe_data(dataset,'is_genuine')
#         input('\n...\n')
#         nut_preprocessing.compute_vif(matrix,feature)
#         nut_exploration.plot_describe(dataset,'is_genuine',plot_path=PLOT_PATH)
#     
#     elif sys.argv[1] == '--describe-model':
#         clear_screen()
# 
#         if (len(sys.argv) < 3) or sys.argv[2] not in ['expit','kmeans']:
#             raise ValueError('Utilisation : --describe-model [nom_du_model : expit / kmeans]')
# 
#         gen_preprocessor = nut_preprocessing.build_preprocessor(matrix,target,plot_path=PLOT_PATH)
#         nut_modeling.evaluate_model(gen_preprocessor,model=sys.argv[2],plot_path=PLOT_PATH)
# 
#     elif sys.argv[1] == '--build-model':
#         clear_screen()
# 
#         if (len(sys.argv) < 3) or not os.path.isdir(sys.argv[2]):
#             raise FileNotFoundError('Vous devez fournir un dossier d\'exportation')
# 
#         model_path = sys.argv[2]
# 
#         # Split data into train and test sets
#         x_train, x_test, y_train, y_test = train_test_split(matrix, target, stratify=target)
#         nut_modeling.build_model(x_train, y_train, model_path)
# 
#         if (len(sys.argv) == 4) and (sys.argv[3] == '--eval'):
#             nut_evaluating.compare_model(x_test,y_test,feature, model_path, plot_path=PLOT_PATH)
# 
#     elif sys.argv[1] == '--use-model':
#         clear_screen()
# 
#         if (len(sys.argv) < 4) or not os.path.isdir(sys.argv[2]):
#             raise FileNotFoundError('Chemin incorrect ou dataset absent')
# 
#         model_path = sys.argv[2]
# 
#         for file in ['expit','kmeans']:
#             if not os.path.exists(f'{model_path}/{file}_model.pkl'):
#                 x_train, x_test, y_train, y_test = train_test_split(matrix, target, stratify=target)
#                 nut_modeling.build_model(x_train, y_train, model_path)
#                 break
# 
#         if not os.path.exists(sys.argv[3]):
#             raise FileNotFoundError('Fichier introuvable')
# 
#         data_file = sys.argv[3]
#         nut_evaluating.use_model(data_file,'id',model_path)
# 
#     else:
#         raise ValueError('Paramètres incorrects < --help > pour afficher l\'aide')
# 
# if __name__ == '__main__':
#     if len(sys.argv) <= 1:
#         sys.exit('Trop peu d\'arguments pour exécuter le script')
#     main()
