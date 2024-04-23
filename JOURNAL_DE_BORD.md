# JOURNAL DE BORD

## Jour 1:
Déroulement de la journée :
  - répartition des tâches :
    - Statistique sur les données
    - création du model en pytorch
    - recherche explicabilité + matrice de confusion
    - recherche sur les hyperparamètres
  - creation d'un premier model avec 65% d'accuracy sur le dataset de test et 93% pour le dataset de train

Nous avons constaté que les classes n'étaient pas équilibrées et que les images ne sont pas de la même taille. Nous utilisons un système de poids à l'aide de compute_class_weight pour faire en sorte que le modèle prête plus d'attentions aux classes les moins représentées et donc régler le problème de distribution non équilibrée. Toutes les images ont été redimensionnées: à l'heure actuelle la taille recommandée par la documentation est utilisée mais nous planifions de la changer après avoir réalisé des études.

Après un premier lancement, nous constatons un overfitting de notre premier model sur les donnéees de train. Des modifications des hyperparamètres sont prévues pour régler ce problème.

Le nombre élévé de classes rend difficile la lecture de la matrice de confusion. Nous planifions de la split en plusieurs parties pour résoudre ce problème.
