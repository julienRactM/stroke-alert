# Projet de Data Engineering et Data Science de la plateforme  **stroke-alert**.

## Informations importantes à l'appréciation du projet :
- pour le script .py de l'exécution de la prédiction
- Le requirement.txt est conséquent du fait que nous sommes 3 à travailler sur le projet avec des librairies variés et utilisons flask.



## Dévelopement d'un outil d'aide à la prévention d'AVC.

L'outil aura pour mission d'effectuer une prédiction des risques pour un ou plusieurs individus de subir un accident vasculaire cérébral à partir de critères médicaux simplifiés


### Sont mis à notre disposition :

#### ● Des statistiques sur les accidents vasculaires cérébraux en France

Chiffres clés concernant les accidents vasculaires cérébraux en France :
En 2021, 121 940 personnes ont présenté un AVC aigu (52,6 % étaient des
hommes et 47,4 % des femmes).
Parmi les patients pris en charge pour un AVC aigu, certains présentaient une
ou plusieurs autres maladies cardiaques ou facteurs du risque
cardiovasculaire :
- 34 % ont des troubles du rythme cardiaque ;
- 23 % ont une forme de diabète ;
- 17 % une maladie coronaire chronique ;
- 8% une artériopathie des membres inférieurs ;
- 8% une maladie des valves cardiaques.

En 2021, 849 300 personnes ont été suivies pour des séquelles d'AVC (51,8 %
étaient des hommes et 48,2 % des femmes).
L'AVC est la première cause de mortalité chez la femme avant le cancer du
sein et la troisième chez l'homme.

Il est la première cause de handicap acquis de l’adulte et la deuxième cause
de démence après la maladie d'Alzheimer.

#### ● Une base de données contenant les variables suivantes :

**Explication des données**
- id : identifiant unique.
- gender : genre du patient.
- age : âge du patient.
- hypertension : 0 si le patient n'a pas d'hypertension, 1 si le patient a de
l'hypertension.
- heart_disease : 0 si le patient n'a pas de maladie cardiaque, 1 si le
patient a une maladie cardiaque.
- ever_married : si le patient a déjà été marié ou pas.
- work_type : type de travail du patient.
- Residence_type : type de résidence du patient "Rural" ou "Urbain".
- avg_glucose_level : taux moyen de glucose dans le sang.
- bmi : indice de masse corporelle.
- smoking_status : "a déjà fumé", "n'a jamais fumé", "fumé" ou "Inconnu ".
- stroke : 1 si le patient a eu un accident vasculaire cérébral ou 0 s'il n'en
a pas eu.

### Résumé de l'Analyse Exploratoire :

**Premières observations :**

- **Il est important de noter que parmis les 5110 individus qui composent le jeu de données, seul 249 d'entre eux ont déjà subi un AVC, soit moins de 5% de la population que présente le jeu de données initial.**

- Seulement 3 personnes sur les 2024 individus de moins de 38 ans ont subis un avc dans notre jeu de données, il pourrait être intéressant d'exclure ces 2024 individus du jeu de données pour faciliter le rééquilibrage des classes toute en excluant une partie de la population très peu concernée par les risques d'AVC. Les models associeraient rapidement la relation croissante entre l'âge et les risques d'AVC qui est déjà prouvé scientifiquement et reconnu depuis des siècles :

A partir de 55 ans, chaque décénie double le risque d'AVC.

D'après :
Rothwell PM, Coull AJ, Silver LE, et al. Population-based study of event-rate, incidence, case fatality, and mortality for all acute vascular events in all arterial territories (Oxford vascular study). Lancet 2005; 366 : 1773–83.
- Très peu de valeurs manquantes à l'exception de la variable smoking_status (30%) et dans une moindre mesure bmi (3.93%).
- On ne peut pas se séparer des valeurs manquantes de la feature bmi puisque l'on perdrait 40 de nos 249 patients ayant déjà subi un avc.
- L'échantillon comprend des individus dont l'âge est compris entre 8 mois et 82 ans (moyenne = 43 ans).
- La colonne avg_glucose_level semble contenir des données pertinentes et sans outliers.
- On remarque qu'il existe 3 classes dans la colonne genre, l'une d'entre elle n'ayant qu'une seule entrée, on décide de s'en séparer.
- Après vérification, il apparaît que la colonne bmi contient des valeurs abbérantes car il n'est pas possible d'avoir un IMC inférieur à 10.3 ou supérieur à 97.60. On retire les entrées qui contiennent des valeurs inférieures à 10 et supérieures à 50


On analyse séparemment les individus ayant subi un avc et ceux qui n'ent ont jamais eu, on en tire ces conclusions :

**Observations des données des personnes classés 0 (pas d'avc):**
* On compte 60% de femmes pour 40% d'hommes
* La colonne âge indique un score skewness de -0.08 et un score Kurtosis de -0.98, ce qui nous permet de supposer que malgré une légère asymétrie à gauche et un léger aplatissement par rapport à une distribution normale, cette distribution est relativement proche de la normale.
* 92% des personnes de ce groupe ne souffrent pas d'hypertension
* 95% des personnes de ce groupe n'ont pas de maladies cardiaques.
* 64% des personnes ont déjà été mariés.
* 57% des personnes travaillent dans le secteur privé
* Il n'y a pas de classe majoritaire dans la colonne residence type
* Pour la colonne avg glucose level, les valeurs indiquent une distribution qui est à la fois asymétrique vers la droite (skewness positif) et a des pics plus élevés par rapport à une distribution normale (kurtosis positif).
* La colonne bmi indique une asymétrie à droite (présence d'outliers) et une distribution plus élevée que la normale.


**Observartions des données des personnes classés 1 (avc):**
* La colonne 'gender' comprend 56% de femmes.
* La colonne âge indique une asymétrie à droite avec un skewness à -1.37 et un applatissement plus haut que pour une distribution normale.
* 73% des personnes de ce groupe n'ont pas d'hypertension
* 81% des personnes de ce groupe n'ont pas de maladies cardiaques
* 88% des personnes ont déjà été mariés
* 59% des personnes travaillent dans le secteur privé
* 55% des personnes vivent en région urbaine
* La colonne avg_glucose est de type bimodale ce qui sous entend qu'il y a 2 sous ensembles au sein de cette catégorie (une concentration entre 50 et 150 et une autre entre 150 et 250)
* Pour la colonne bmi, les valeurs suggèrent une distribution légèrement asymétrique vers la droite, avec une concentration de valeurs plus importante autour de la moyenne et des extrémités plus épaisses que dans une distribution normale. Le score du kurtosis relativement faible indique que la distribution a des pics qui ne sont pas très élevés par rapport à une distribution normale.
* 36% des personnes de ce groupe n'ont jamais fumé et 28% sont des anciens fumeurs.

Nos deux groupes présentent des caractéristiques différentes. C'est pour cette raison que l'on a choisi d'utiliser la méthode d'imputation KNN Imputer pour la colonne BMI.

## Imputation

On opte pour utiliser KNN Imputer sur la variable BMI et on choisit une imputation hybride des valeurs manquantes de la colonne smoking_status, on part du postulat que les personnages mineurs sont non fumeurs, puis on





## Encodage

**Définition: un encodeur est utilisé pour convertir des variables catégorielles en une variable numérique afin de pouvoir les utiliser sur des modèles de machine learning, qui travaillent généralement avec des données numériques.**.

Il existe différents type d'encodeurs:

**LabelEncoder:**

Il convertit chaque classe unique d'une variable catégorielle en un nombre entier. Utile lorsque l'ordre des catégories n'a pas d'importance.

**OneHotEncoder:**

Il crée une colonne binaire distincte pour chaque catégorie unique. Utile lorsque l'ordre des catégories n'a pas d'importance.

**OrdinalEncoder:**

Il ressemble au LabelEncoder, mais avec la possibilité de spécifier un ordre explicite des catégories.

**Dans notre cas nous aurons uniquement besoin du OneHotEncoder pour la feature smoking_status.**



## Standardisation (normalisation)

La standardisation des données, également appelée *normalisation*, fait référence au processus de transformation des données brutes en une forme standardisée. La plupart du temps, cela implique de procéder à la modification des données afin que ces dernières obtiennent **une moyenne de zéro et un écart-type de un**.

Sklearn propose 3 types de standardiseurs :
* **Standard Scaler** : Il standardise les données en les centrant autour de zéro (moyenne = 0) et en les mettant à l'échelle en fonction de l'écart type.
* **Min Max Scaler** : Il met à l'échelle les données dans une plage spécifique, généralement entre 0 et 1. Il est utile lorsque les données ont une distribution non normale ou avec des algorithmes sensibles à l'échelle.
* **Robust Scaler** : Il utilise des statistiques robustes en éliminant les médianes et en échelonnant les données en fonction des quantiles. Cela le rend robuste aux valeurs aberrantes.

**On détermine que le Min Max Scaler correspond à notre cas d'usage, ayant décider de nous séparer des outliers sur les variables que l'on compte utiliser**

On détermine que le Robust Scaler correspond à notre cas d'usage, il est très flexible et la distribution de nos données n'est pas normale.


### Sélection des Features

On réalise des tests anova et de khi2 afin de déterminer quelles variables sont corrélées avec la variable cible "stroke".


### Modelisation

Nos données sont en fin exploitable pour un algorithme de classification supervisée
**Explication de la classification supervisée**
La classification supervisée est un type d'apprentissage supervisé où l'algorithme est entrainé sur un dataset dit labélisé, la caractéristique(feature) que l'on va chercher à prédire doit déjà exister dans le jeu de données pour l'appliquer.

C'est donc un processus fondamentalement différent de l'apprentissage non supervisé où la caractéristique que l'on souhaite prédire n'existe pas dans le jeu de données.


#### Modelisation sur une seule feature

Ne devant utiliser qu'une seule feature nous choisissons d'utiliser l'âge pour la régression logistique et le random forest aux dépens de l'hypertension/maladies cardiaques et nous choisissons bmi pour le svm.

nous avons décider d'utiliser les models suivants:
* **regression logistique**
La régression logistique est un modèle statistique permettant d’étudier les relations entre un ensemble de variables qualitatives Xi et une variable qualitative Y.
Il s’agit d’un modèle linéaire généralisé utilisant une fonction logistique comme fonction de lien.

![alt text](image-2.png)

* **random forest**
Random forest signifie « forêt aléatoire », c’est un algorithme qui se base sur l’assemblage d’arbres de décision. Il est assez intuitif à comprendre, rapide à entraîner et il produit des résultats généralisables.

![alt text](image-1.png)

* **SVM**
C'est une famille d'algorithmes d'apprentissage automatique qui permettent de résoudre des problèmes tant de classification que de régression ou de détection d'anomalie.

![alt text](image-3.png)

#### Conclusion sur les résultats des models en n'utilisant qu'une seule feature :

Nos résultats sont déjà prometteurs avec une seule feature, notamment pour le random forest et la régression logistique qui ont un rapport précision/recall intéressant tandis que le SVM est excellent pour prévoir les avc avérés mais prédit un nombre bien trop important d'individus n'ayant pas subis d'AVC comme étant susceptible d'en subir un.


#### Modelisation sur plusieurs features

On choisit de conserver le model random forest et d'y rajouter les features que nous considérons impactantes lors de nos précédentes observations, c'est-à-dire l'âge, l'hypertension, les niveaux de glucose moyens, les maladies cardiaques et le status de fumeur

![alt text](image-5.png)



#### Conclusion sur les résultats des models complets :
