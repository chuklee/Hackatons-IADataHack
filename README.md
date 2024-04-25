# Hackathon 2024 : Challenge 6milarité

## Auteurs:
- Florine Kieraga
- Eliana Junker
- Martin Natale
- Eithan Nakache
- Sacha Hibon

## Architecture Git
- JOURNAL_DE_BORD : journal de board expliquant les démarches et résultats
- requirements.txt: contient les librairies nécessaires. Avant d'explorer les notebooks, il est conseillé de les installer (faisable avec la commande "pip install -r requirements.txt" dans un terminal)
- notebooks (à lancer dans cet ordre): 
  - telechargement_donne.ipynb : permet de récupérer toutes les données qui sont sur AWS3
  - analyse_donnee.ipynb : permet d'obtenir plusieurs statistiques sur les données d'origine
  - entrainement.ipynb : permet de créer et entraîner notre modèle (le générique sur les données d'origine ou le spécifique sur les classes spécifiées (pour cela, passer use_subset à True))
  - evaluation_model.ipynb : permet de voir les études des résultats sur un modèle spécifique (matrice de confusion et explicabilité)
- helper : contient toutes les fonctions pythons implémentées par nos soins et utilisées par les notebooks

## Tester avec différents dataset

Si vous voulez jouer un peu avec nos modèles, vous pouvez lancer les fichiers python
'blur_transformation.py' et 'chargement_donnee.py' dans helper qui créeront respectivement des datasets. Un qui ajoutera des données floutées, l'autre qui créera un dataset réparti tel que 80% train 20 test.

Une fois les dataset créés, vous avez seulement à exécuter une nouvelle fois le notebook 'entraînenement' et 'evaluation_model' pour comparer les performances.