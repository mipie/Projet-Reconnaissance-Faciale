Description de la fonctionnalité de reconnaissance faciale

Cette fonctionnalité repose sur un réseau siamois conçu pour comparer deux images et déterminer si elles représentent la même personne. Elle est utilisée pour des tâches d’authentification faciale en exploitant des techniques avancées de vision par ordinateur et d’apprentissage profond.

Le système commence par prétraiter les images, en redimensionnant et normalisant chaque entrée pour assurer une cohérence dans l’apprentissage du modèle. Il extrait ensuite des caractéristiques uniques de chaque visage à l’aide d’un réseau de convolution (CNN) composé de plusieurs couches de convolution et de pooling.

Les images d’ancrage et de validation sont ensuite comparées via une couche personnalisée, L1Dist, qui calcule la distance absolue entre leurs vecteurs d’embedding. Un classificateur final, utilisant une activation sigmoïde, estime la probabilité que les deux images appartiennent à la même personne.

Le modèle est entraîné avec une perte binaire cross-entropie, optimisée grâce à l’algorithme Adam. Des métriques comme la précision et le rappel sont utilisées pour évaluer la performance du modèle sur des ensembles de test.

Enfin, le modèle entraîné est sauvegardé et peut être rechargé pour effectuer des prédictions sur de nouvelles images, facilitant ainsi son intégration dans des applications nécessitant une identification ou une vérification faciale en temps réel.

