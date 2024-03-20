# Organisation du Projet

Le projet est organisé comme suit :

- **main.py:** Fichier principal du projet.
- **presentation.pdf** : Présentation du projet.
- **conf:** Répertoire contenant les fichiers de configuration.
- **file:** Répertoire contenant les données originales.
- **notebook:** Répertoire contenant les notebooks du projet.
- **pkl:** Répertoire contenant les données sérialisées.
- **plot:** Répertoire contenant les exports des graphiques.
- **src:** Répertoire contenant les modules du projet.
- **style:** Répertoire contenant les fichiers de style HTML et Matplotlib.

**Note:** Pour des raisons d'espace de stockage, la base de données ```geopandas``` est désactivée.

# Description du Jeu de Données

## Aperçu

Ce jeu de données fournit des informations sur les métriques liées à la volaille, les dépendances commerciales et les indicateurs socio-économiques pour différents pays. Les données couvrent divers aspects tels que la production de volaille, les importations, les exportations, les taux d'autosuffisance et des indicateurs clés liés à la population et aux facteurs économiques.

## Dictionnaire des Données

| Caractéristique                           | Description                                          | Unité de Mesure                           | Calcul                                              |
|-------------------------------------------|------------------------------------------------------|-------------------------------------------|------------------------------------------------------|
| `Zone`                                    | Index du pays                                        | N/A                                       | N/A                                                  |
| `Production Viande de Volailles`           | Production de viande de volaille                     | Milliers de tonnes                        | N/A                                                  |
| `Importations Viande de Volailles`         | Importations de viande de volaille                   | Milliers de tonnes                        | N/A                                                  |
| `Exportations Viande de Volailles`         | Exportations de viande de volaille                   | Milliers de tonnes                        | N/A                                                  |
| `TDI Volailles`                            | Indice de Dépendance aux importations (TDI) pour la volaille | Sans unité                               | $\text{{TDI}} = \frac{{\text{{Importations viande de volaille}}}}{{\text{{Disponibilité intérieure de Viande de Volailles}}}}$           |
| `Ratio d'autosuffisance volailles`         | Taux d'autosuffisance pour la volaille               | Sans unité                               | $\text{{Ratio d'Autosuffisance Volailles}} = \frac{{\text{{Production Viande de Volailles}}}}{{\text{{Disponibilité intérieure de Viande de Volailles}}}}$             |
| `Ratio Volailles / viande`                 | Quantité de viande de volaille sur l'ensemble des viandes | Sans unité                            | $\text{{Ratio}} = \frac{{\text{{Quantité de volaille}}}}{{\text{{Quantité totale de viande}}}}$   |
| `Disponibilité totale viande`              | Disponibilité totale de viande                       | kg par personne par jour                | N/A                                                  |
| `Disponibilité alimentaire`                | Disponibilité alimentaire totale                     | kcal par personne par jour               | N/A                                                  |
| `Population`                              | Population du pays                                   | Unité                                    | N/A                                                  |
| `PIB`                                     | Produit Intérieur Brut (PIB)                         | Euros                                   | $\text{{PIB}} = \text{{PIB (en dollars)}} * \text{{Facteur de conversion en Euros 2017}}$ |
| `PIB par habitant`                        | Produit Intérieur Brut par habitant                  | Euros                                   | $\text{{PIB par habitant}} = \frac{{\text{{PIB}}}}{{\text{{Population}}}} * \text{{Facteur de conversion en Euros 2017}}$ |

Note : Le facteur de conversion de dollars à euros en 2017 est de 0,8867.

## Unités de Mesure

- `Milliers de tonnes` représente une quantité en milliers de tonnes.
- `Sans unité` indique une quantité sans dimension.
- `kg par personne par jour` représente des kilogrammes de viande par personne et par jour.
- `kcal par personne par jour` représente des kilocalories de nourriture par personne et par jour.
- `Euros` représente la monnaie en euros.

## Source des Données

Les données proviennent de la [FAO](https://www.fao.org/faostat/fr/#data).

## Licence

Ce jeu de données est fourni sous [CC BY-NC-SA 3.0 IGO](https://creativecommons.org/licenses/by-nc-sa/3.0/igo/).

Le Ratio d'Autosuffisance en volaille est calculé à l'aide de la formule suivante :
