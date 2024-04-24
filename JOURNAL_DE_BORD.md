# JOURNAL DE BORD

## Introduction:
Dans ce journal de bord, nous allons expliqué les différentes étapes de notre travail au fil des jours.  

## Jour 1:
Répartition des tâches :
  - Statistique sur les données
  - Création du model en pytorch
  - Recherche sur l'explicabilité + matrice de confusion
  - Recherche sur les hyperparamètres
  
### Statistique sur les données

#### Répartition des classes

##### Toutes classes confondues

![alt text](images/all_classes_plot.png)
![alt text](images/all_classes_metrics.png)

Conclusion : les données dans les classes sont plutôt bien réparties.

##### Par année

![alt text](images/classes_by_year.png)

Conclusion : les données dans les classes ne sont pas bien réparties par année.

##### Par marque (premier mot de chaque classe)

![alt text](images/classes_by_model.png)

Conclusion : les données dans les classes ne sont pas bien réparties par marque.

#### Analyse sur la taille des images

##### Graphe

![alt text](images/image_size_graph.png)

##### Table

![alt text](images/image_sizes_metrics.png)

Conclusion : les images sont rectangulaires et leurs tailles généralement de tailles 700X483.

#### Analyse sur la taille des voitures comparées aux images

##### Graph

![alt text](images/car_size_comparing_image_size_graph.png)

##### Table

![alt text](images/car_size_comparing_image_size_metrics.png)

### Création du model en pytorch
Nous avons fait le choix d'utiliser pytorch pour l'entrainement de notre modèle. Cette bibliothèque à l'avantage d'être très flexible et permet d'importer facilement le modèle resnet18.

#### Premier version du modèle
Comme remarqué dans les statistiques des données le dataset de base est répartit en 50% de train et 50% de test. Les seuls transformations effectué sont la normalisation et la transformation en tensor. Après avoir demandé à un coach, il nous a été expliqué que pour des images RGB, la normalisation basique est la division par 255. Nous avons décidé d'utiliser cette méthode de normalisation.

##### Hyperparamètres
Pour l'optimizer, nous avons utilisé l'optimizer Adam avec un learning rate de 0.001. et une loss function de type CrossEntropyLoss. Voulant testé rapidement la convergence de notre modèle nous utilisons un batch size de 64 qui privélégie la rapidité d'apprentissage pour délaisser les perfomances du modèle.
Nous avons constaté que la dernière couche de base de resnet18 posède une sortie prévoyant 1000 classes. Afin que le modèle puisse prédire nos classes, nous avons changé la couche de sortie en une couche de 196 neurones. Nous avons ensuite entrainé le modèle sur 10 epochs.

##### Résultats
Le modèle a atteint une accuracy de 65% sur le test. Nous avons donc décidé de changer les hyperparamètres pour voir si nous pouvions améliorer les performances du modèle.

##### Explication du résultat
Les performances du modèle sont faibles.
Cela peut être dû à plusieurs facteurs :
- Le modèle underfit et n'a pas eu le temps de converger
- Les hyperparamètres ne sont pas adaptés
- Le set de données est trop petit

Nous allons donc essayer de changer les hyperparamètres pour voir si nous pouvons améliorer les performances du modèle.

#### Deuxième version du modèle
##### Hyperparamètres
Afin d'améliorer les performances du modèle, nous avons décidé d'utiliser une méthode de data augmentation sur le jeu de train. Les transformations effectuées sont les suivantes : RandomHorizontalFlip, qui vas inverser certaines images et RandomRotation(15) qui vas faire une rotation de 15 degrés sur certaines images. Nous ajoutons également un learning rates scheduler de type ReduceLROnPlateau avec un facteur de 0.1 et une patience de 5. Nous avons également décidé d'utiliser un batch size de 32 pour améliorer les performances du modèle. Nous avons ensuite entrainé le modèle sur 10 epochs.

##### Résultats
Le modèle a atteint une accuracy de 0.98 sur le train et de 0.88 sur le test. Les performances du modèle ont été grandement améliorées. Nous remarquons néanmoins qu'une divergence commence à apparaitre entre les performances du train et du test, ce qui peut indiquer un overfit du modèle.

### Recherche sur l'explicabilité
Nous avons débuté des recherches sur l'explicabilité afin de comprendre en quoi cela consistait et quelles bibliothèques étaient disponibles.

Lors de ces recherches, nous avons trouvé plusieurs bibliothèques disponibles pour les modèles PyTorch :
  - SHAP (SHapley Additive exPlanations)
  - Captum
  - Torch-Cam qui permet un visualisation avec la génération des cartes d'activations.
  - Lime

Nous avons découvert différentes techniques d'explicabilité :
  - SHAP : basée sur la théorie des jeux, permet de mesurer l’impact sur la prédiction d’ajouter une variable grâce à des permutations de toutes les options possibles.
  - Lime (Local Interpretable Model-agnostic Explanations): explique les prédictions des modèles en ajustant un modèle interprétable local autour de l'instance à expliquer.
  - Occlusion : cette méthode implique la modification de parties de l'image d'entrée pour évaluer l'impact sur la sortie du modèle.
  - Grad-CAM et CAM (Classe Activation Mapping) : génèrent des cartes de chaleur pour visualiser les régions importantes de l'image qui ont contribué à la prédiction de classe du modèle.

Nous avons décidé d'utiliser la librairie Captum pour l'occlusion et la librairie Torch-Cam pour le Grad-Cam

Sources:
 - https://medium.com/@mariusvadeika/captum-model-interpretability-for-pytorch-c630fadfc6be 
 - https://github.com/frgfm/torch-cam
 - https://github.com/pytorch/captum 
 - https://indabaxmorocco.github.io/materials/hajji_slides.pdf
 - https://aqsone.com/blog/articlescientifique/interpretabilite-et-explicabilite-des-modeles-de-machine-learning/
 - https://courses.minnalearn.com/en/courses/trustworthy-ai/preview/explainability/types-of-explainable-ai/
 
### Recherche sur les hyperparamètres

Afin d'améliorer les performances de notre modèle nous avons décidé de faire une recherche sur les hyperparamètres suivants :

- Batch size : Le batch size représente le nombre d'échantillons qui seront propagés à travers le réseau de neurones. Nous avions décidé d'explorer les batch sizes suivants : `16`, `32`, `64`, `128`.

- Learning rate : Le learning rate, car il était crucial de déterminer le taux d'apprentissage optimal pour la convergence du modèle. Les valeurs que nous avions décidé d'explorer étaient : `0.0001` et `0.001` .

- Optimizer : Le choix de l'optimizer est crucial pour la convergence du modèle. Il permet de mettre à jour les poids du réseau de neurones. Nous avions décidé d'explorer les optimizers suivants : `Adam`, `SGD`.

- Scheduler : Le scheduler permet de modifier le learning rate au cours de l'entraînement. Nous avions décidé d'explorer les schedulers suivants : `StepLR`, `ReduceLROnPlateau`.

- Poids des classes : Nous avions décidé de d'analyser la distribution des classes pour déterminer si nous devions utiliser des poids de classe. Nous avions décidé d'explorer les poids des classes suivants à l'aide de la librairie `sklearn` : `None`, `balanced`.

- Normalisation : La normalisation des données est une étape importante pour l'entraînement du modèle. Nous avions décidé d'explorer les normalisations suivantes : `None`, `ImageNet`, `Z-score`.

- Data augmentation : Nous avions décidé de mener une recherche sur la data augmentation, car il était important de trouver la data augmentation qui permettrait au modèle de converger. Nous avions choisi d'explorer les data augmentations suivantes : `None`, `HorizontalFlip`, `VerticalFlip`, `RandomRotation` , `AutoAugmentation(ImageNet)`.

- Image Shape : Nous avions décidé de mener une recherche sur l'image shape, car il était important de trouver la bonne forme d'image pour que le modèle converge. Nous avions choisi d'explorer les formes d'image suivantes : `224`, `256`, `512`, `672`.

Sources: 
- https://optuna.readthedocs.io/en/stable/index.html
- https://pytorch.org/docs/stable/optim.html
- https://scikit-learn.org/stable/modules/generated/
- https://pytorch.org/vision/stable/transforms.html
- https://pytorch.org/vision/stable/models.html
- https://pytorch.org/docs/stable/nn.html

### Résultats des hyperparamètres

- Learning rate : `0.0001` avec Adam, on obtenait de meilleur resultat comparé à `0.001`. 

- Poids des classes : l'utilisation des poids n'a pas été adopté car nous avons remarqué que les classes étaient équilibrés entre elles. Cela entrainait également un problème au niveau de l'entrainement du modèle.

- Normalisation : nous avons utilisé la normalisation [0.5,0.5,0.5]

- Data augmentation : `HorizontalFlip`, `RandomRotation`: nous gardons ceux-ci car ils donnent de meilleurs résultats  mais nous n'utiliserons pas `AutoAugmentation(ImageNet)` car celui-ci détruit les performances du modèle (baisse l'accuracy sur test à 10%).

### Conclusion de la journée
Nous avons réussi à entraîner un modèle avec une accuracy de 0.88 sur le test. Nous avons également commencé à explorer les différentes méthodes d'explicabilité pour comprendre comment notre modèle prend ses décisions. Nous avons également commencé à explorer les hyperparamètres pour améliorer les performances de notre modèle. Nous sommes globalement satisfaits de notre journée et nous avons hâte de continuer à explorer les différentes méthodes d'explicabilité et d'optimisation de notre modèle.

## Jour 2:

### Résulats sur les hyperparamètres
- Batch size : Le batch size représente le nombre d'échantillons qui seront propagés à travers le réseau de neurones. Nous avions décidé d'explorer les batch sizes suivants : `16`, `32`, `64`, `128`.
- Optimizer : Le choix de l'optimizer est crucial pour la convergence du modèle. Il permet de mettre à jour les poids du réseau de neurones. Nous avions décidé d'explorer les optimizers suivants : `Adam`, `SGD`.
- Scheduler : Le scheduler permet de modifier le learning rate au cours de l'entraînement. Nous avions décidé d'explorer les schedulers suivants : `StepLR`, `ReduceLROnPlateau`.

- Image Shape : Nous avions décidé de mener une recherche sur l'image shape, car il était important de trouver la bonne forme d'image pour que le modèle converge. Nous avions choisi d'explorer les formes d'image suivantes : `224`, `256`, `512`, `672`.

### Explicabilité
Nous avons commencé à chercher à expliqué les performances de notre model et comment nous pourions améliorer celui-ci.
Nous avons choisi de regarder l'occlusion et le Grad-Cam pour les images qui était mal détécté par le modèle. Les images mal détectés étaient dans la catégorie des Audi.
