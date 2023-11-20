# État de l'art

Globalement, les CNNs sont utilisés pour deux types de tâches:

- **Classification** : Dans une tâche de classification, le CNN attribue une classe à l'entrée en produisant une sortie qui représente la probabilité d'appartenance de l'entrée à chaque classe prédéfinie.

- **Régression** : D'autre part, les CNN peuvent être utilisés pour des tâches de régression où la sortie n'est pas une classe, mais plutôt un vecteur de valeurs continues.

Notre objectif étant d'associer des coordonnées GPS à une image, nous allons donc faire de la **régression**.

## Étude des modèles

Voici des modèles CNN qui pourraient être utilisés pour faire de la régression:
- [LeNet-5](https://medium.com/codex/lenet-5-complete-architecture-84c6d08215f9)
- [AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
- [VGG-16/VGG-19](https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918)
- [ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
- [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/)

### LeNet-5

Taille d'image en entrée fixée à 28x28 pixels, car ce modèle est principalement utilisé pour la reconnaissance de caractères.

### AlexNet

Amélioration du LeNet. Taille d'image en entrée 224x224.

### VGG16/VGG19

Input à 224x224. Choix entre de modèles à 16 ou 19 couches.

### ResNet

Input variable. Près de 34 couches, mais adaptable. 

### DenseNet

Contrairement aux autres qui utilisent des couches appliquant des convolutions, ce réseau est basé uniquement sur une succession de couches denses. Il prend également en entrée des images d'au moins 224x224.

## Étude d'une solution proposée

Étudions les outils utilisés par une [solution](https://nirvan66.github.io/geoguessr.html) proposée.

Les modèles utilisés sont basés sur du ResNet et du LSTM. Les images en entrée sont en 300x300 pixels.