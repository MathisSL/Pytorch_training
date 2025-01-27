# Pytorch_training
Creating a Pytorch model to predict class of images of the MNIST dataset, dans ce readme, je vais expliquer la structure du modèle créé.

## nn.Conv2d
La classe torch.nn.Conv2d est utilisée dans PyTorch pour appliquer une convolution 2D, une opération clé dans les réseaux de neurones convolutifs (CNN).

### Calculs et sortie après appliquation de filtres convolutifs

![conv2d_gif](https://github.com/user-attachments/assets/aa333ec2-bbe5-4f2d-b0ac-e827382c8dbf)


![Conv2d_1](https://github.com/user-attachments/assets/dd9cea7f-c5a3-4307-8165-b81208794b6f)
![Conv2d_2](https://github.com/user-attachments/assets/8a0a12d3-d04f-4f46-9e60-1306933c6a96)

On a :
  - Filtres : détectent des motifs (bords, textures, etc.).
  - Convolution : calcule des nouvelles représentations basées sur ces motifs.
  - Paramètres comme stride et padding contrôlent la taille et la précision de la sortie.

## BatchNorm2d

La classe torch.nn.BatchNorm2d est utilisée pour normaliser les activations (sorties) d'une couche convolutive sur un lot (batch) pendant l'entraînement. Cela aide à accélérer l'entraînement et à stabiliser le réseau.
![BatchNorm2d_img](https://github.com/user-attachments/assets/4a88d189-9be9-448c-852a-9df69b828d89)

### Rôle de nn.BatchNorm2d :

Après une couche de convolution, les activations peuvent avoir des valeurs très différentes entre les images d'un lot ou même entre les différents canaux. BatchNorm2d :
    Normalise les activations pour qu'elles aient une moyenne proche de 0 et une variance proche de 1.
    Ajoute ensuite des paramètres apprenables pour ajuster ces valeurs normalisées (échelle et décalage).
![BN_1](https://github.com/user-attachments/assets/1079ecb2-b124-44f9-acab-9ade1039f7b9)

![BN_2](https://github.com/user-attachments/assets/892a894a-4cbe-4ad9-985c-c5839a1dc3e1)

Entrée : Sorties d'une convolution (dimensions batch × canaux × hauteur × largeur).
Traitement :
    Calcul de la moyenne et de la variance par canal sur tout le lot.
    Normalisation pour avoir des activations stables.
    Réglage avec des paramètres apprenables.

Utilité : Stabilise et accélère l'entraînement.

## nn.ReLU

La fonction d'activation ReLU pour Unité linéaire rectifiée introduit de la non-linéarité dans le réseau. Cela permet aux réseaux de modéliser des relations complexes entre les données (contrairement aux fonctions linéaires qui ne peuvent modéliser que des relations simples).
Elle aide à éviter le problème de gradients qui disparaissent (vanishing gradient problem) observé avec des fonctions comme sigmoid ou tanh.

![ReLU](https://github.com/user-attachments/assets/91c59120-74bf-486f-ab74-2b49e89622f8)


Il ne faut pas confondre :

    torch.nn.ReLU :
        C'est une classe utilisée comme une couche dans un modèle.
        Exemple : nn.ReLU(inplace=True).

    torch.relu :
        C'est une fonction directe. On peut l'appliquer directement à un tenseur.
        Exemple : torch.relu(tensor).


Le paramètre inplace (dans nn.ReLU) contrôle si l'opération modifie le tenseur d'entrée directement ou crée un nouveau tenseur :

    inplace=True :
        Modifie directement les valeurs dans le tenseur d'entrée.
        Économise de la mémoire, mais peut causer des erreurs si l'entrée est encore utilisée ailleurs dans le calcul.

    inplace=False :
        Crée un nouveau tenseur pour stocker la sortie (par défaut).

Donc pour conclure sur nn.ReLu :
  Rôle : Rend les sorties non-linéaires en supprimant les valeurs négatives (x<0→0x<0→0).
  Avantages :

    Simple et rapide à calculer.
    Évite le problème de gradients qui disparaissent.

Inconvénients : Les neurones avec x≤0 peuvent rester inactifs (problème de "neurones morts").
Utilisation : Ajoutée après une couche convolutive ou linéaire dans les modèles.

## nn.maxPool2d

La couche torch.nn.MaxPool2d est une opération de pooling utilisée dans les réseaux de neurones convolutifs (CNN). Elle réduit la taille des images (ou des cartes de caractéristiques) tout en conservant les informations les plus importantes.
### Rôle de MaxPool2d :

MaxPool2d est une opération de pooling max qui consiste à extraire la valeur maximale dans une fenêtre de taille définie, souvent pour réduire la taille d'une image tout en gardant les caractéristiques importantes.


![maxpool2d_gif](https://github.com/user-attachments/assets/66a644d0-9b72-4120-b1ee-fae7bf5b3199)

Par exemple, pour chaque sous-région de taille 2×22×2 dans une image, MaxPool2d garde la valeur maximale.
2. Paramètres principaux :

    kernel_size :
        Taille de la fenêtre (ou "noyau") sur laquelle l'opération de pooling est effectuée.
        Par exemple, kernel_size=2 signifie une fenêtre 2×2.

    stride (par défaut kernel_size) :
        Pas de déplacement de la fenêtre.
        Par exemple, stride=2 signifie que la fenêtre se déplace de 2 pixels à chaque fois et 1 pixel par pixel.

    padding (par défaut 0) :
        Ajoute des pixels autour de l'entrée avant de réaliser l'opération de pooling. Cela permet de contrôler la taille de la sortie.

    dilation (par défaut 1) :
        Espace entre les éléments dans la fenêtre de pooling. Cela permet d'élargir la fenêtre sans augmenter sa taille.

    ceil_mode (par défaut False) :
        Si True, la sortie aura une taille plus grande (on arrondit vers le haut). Par défaut, l'arrondi est vers le bas.

### Formule pour la taille de la sortie :

La taille de la sortie après un pooling dépend de la taille de l'entrée, de la taille du noyau (kernel_size), du stride et du padding. La formule générale est :

$H_{out}=\[\frac{H−kernel\_size+2⋅padding}{stride}+1\]$

$W_{out}​=\[\frac{W−kernel\_size+2⋅padding}{stride}+1\]$

Où :
    H et W sont la hauteur et la largeur de l'entrée.
    $H_{out}$​ et $W_{out}$​ sont la hauteur et la largeur de la sortie.


