import numpy as np
import pandas as pd
from functools import reduce

def clean_food_availability_data(path, parameter):
    """
    Clean the input dataframe by filling NaN values with the mean.

    Parameters:
    - path (str): Path to the CSV file containing food availability data.
    - parameter (dict): Additional parameters to pass to pd.read_csv.

    Returns:
    pd.DataFrame: Cleaned food availability dataframe.
    """
    food_availability_data = pd.read_csv(path, **parameter)

    # Compute total food availability as a separate dataframe
    mask = food_availability_data.loc[:, 'Élément'] == 'Disponibilité alimentaire (Kcal/personne/jour)'
    total_availability = food_availability_data.loc[mask,['Zone','Valeur']].groupby('Zone').sum()
    # Rename columns
    total_availability.rename({'Valeur': 'Disponibilité alimentaire'}, axis=1, inplace=True)
    
    # Compute meat availability and poultry product ratio
    mask = (food_availability_data.loc[:,'Élément'] == 'Disponibilité alimentaire en quantité (kg/personne/an)') & \
        (food_availability_data.loc[:,'Produit'].str.contains('Viande'))
    meat_availability = food_availability_data.loc[mask,['Zone','Valeur']].groupby('Zone').sum()
    # Rename columns
    meat_availability.rename({'Valeur': 'Disponibilité totale viande'}, axis=1, inplace=True)
    
    # Select relevant data for poultry products
    mask = food_availability_data['Produit'].str.contains('Volaille') &\
        food_availability_data['Élément'].isin(['Production', 'Importations - Quantité', 'Exportations - Quantité','Disponibilité alimentaire en quantité (kg/personne/an)'])
    food_availability_data = food_availability_data.loc[mask]
    
    # Rename elements for readability
    food_availability_data['Élément'] = food_availability_data['Élément'].replace({'Importations - Quantité': 'Importations',
        'Exportations - Quantité': 'Exportations',
        'Disponibilité alimentaire en quantité (kg/personne/an)':'Disponibilité'})
    food_availability_data['features'] = [f"{x} {y}" for x, y in zip(food_availability_data['Élément'], food_availability_data['Produit'])]
    
    # Reshape the dataframe
    food_availability_data.drop(['Élément', 'Produit'], axis=1, inplace=True)
    food_availability_data = food_availability_data.pivot(index='Zone', columns=['features'])
    food_availability_data = food_availability_data.droplevel(0, axis=1)
    food_availability_data.dropna(inplace=True)
    
    # Merge with availability dataframes
    food_availability_data = reduce(lambda left,right : pd.merge(left,right,on='Zone'),
        [food_availability_data,total_availability,meat_availability])

    # Compute TDI (Trade Dependency Index) for poultry
    net_availability = food_availability_data['Production Viande de Volailles'] + food_availability_data['Importations Viande de Volailles'] -\
         food_availability_data['Exportations Viande de Volailles']
    food_availability_data['TDI Volailles'] = food_availability_data['Importations Viande de Volailles'] / net_availability
    
    # Compute self-sufficiency for poultry
    food_availability_data['Ratio d\'autosuffisance volailles'] = food_availability_data['Production Viande de Volailles'] / net_availability
    
    # Compute poultry ratio in meat
    food_availability_data['Ratio Volailles / viande'] = food_availability_data['Disponibilité Viande de Volailles'] / food_availability_data['Disponibilité totale viande']
    
    # Select and reorder relevant columns
    food_availability_data = food_availability_data.loc[:, [
        'Production Viande de Volailles',
        'Importations Viande de Volailles',
        'Exportations Viande de Volailles',
        'TDI Volailles',
        'Ratio d\'autosuffisance volailles',
        'Ratio Volailles / viande',
        'Disponibilité totale viande',
        'Disponibilité alimentaire']]
    
    # Drop rows related to China
    food_availability_data.drop(['Chine - RAS de Hong-Kong', 'Chine - RAS de Macao', 'Chine, Taiwan Province de'], inplace=True)
    
    return food_availability_data

def clean_population_data(path, parameter):
    """
    Clean the input dataframe by filling NaN values with the mean.

    Parameters:
    - path (str): Path to the CSV file containing population data.
    - parameter (dict): Additional parameters to pass to pd.read_csv.

    Returns:
    pd.DataFrame: Cleaned population dataframe.
    """
    population_data = pd.read_csv(path, **parameter)
    
    # Select relevant data for the year 2017
    mask = population_data.loc[:, 'Année'] == 2017
    population_data = population_data.loc[mask]
    
    # Drop the 'Année' column and set 'Zone' as the index
    population_data.drop('Année', axis=1, inplace=True)
    population_data.set_index('Zone', inplace=True)
    
    # Harmonize values and rename the column
    population_data['Valeur'] = population_data['Valeur'] * 10**3
    population_data['Valeur'] = population_data['Valeur'].astype(np.int32)
    population_data.rename({'Valeur': 'Population'}, axis=1, inplace=True)
    
    return population_data

def clean_gdp_data(path, parameter):
    """
    Clean the input dataframe by filling NaN values with the mean.

    Parameters:
    - path (str): Path to the CSV file containing GDP data.
    - parameter (dict): Additional parameters to pass to pd.read_csv.

    Returns:
    pd.DataFrame: Cleaned GDP dataframe.
    """
    gdp_data = pd.read_csv(path, **parameter)
    
    # Set exchange rate for the year 2017
    exchange_rate_2017 = 0.8867
    
    # Rename the column and replace country name for consistency
    gdp_data.rename({'Valeur': 'PIB'}, axis=1, inplace=True)
    gdp_data.replace({'Pays-Bas (Royaume des)': 'Pays-Bas'}, inplace=True)
    
    # Set 'Zone' as the index
    gdp_data.set_index('Zone', inplace=True)
    
    # Convert GDP values to euros
    gdp_data['PIB'] *= exchange_rate_2017 * 10**6
    
    return gdp_data
