# Fonctions pytorch pour la manipulation des tenseurs

1. Gestion des gradients et calcul automatique

    .detach()
    Retourne un nouveau tenseur sans historique de gradients, utile pour éviter de suivre les opérations en mode inference.
    Exemple : x_detached = x.detach()

    .detach_()
    Version in-place de .detach() (modifie le tenseur existant).
    Exemple : x.detach_()

    .item()
    Extrait la valeur scalaire d'un tenseur à un seul élément.
    Exemple : value = tensor.item()

    .requires_grad
    Attribut booléen indiquant si le tenseur suit les gradients.
    Exemple : tensor.requires_grad = True

    .requires_grad_()
    Active/désactive le suivi des gradients (en place).
    Exemple : tensor.requires_grad_(True)

    .grad
    Stocke le gradient du tenseur après un appel à .backward().
    Exemple : print(tensor.grad)

    .backward()
    Calcule les gradients par rétropropagation (utilisé avec les tenseurs scalaires).
    Exemple : loss.backward()

2. Conversion en types Python/NumPy

    .tolist()
    Convertit le tenseur en une liste Python imbriquée.
    Exemple : list_data = tensor.tolist()

    .numpy()
    Convertit le tenseur en un tableau NumPy (partage la mémoire si possible).
    Exemple : np_array = tensor.numpy()

3. Manipulation de la forme (Shape)

    .view()
    Redimensionne le tenseur sans copier les données (forme compatible).
    Exemple : tensor.view(2, 3)

    .reshape()
    Similaire à view(), mais gère automatiquement la contiguïté.
    Exemple : tensor.reshape(2, 3)

    .transpose()
    Permute deux dimensions.
    Exemple : tensor.transpose(0, 1)

    .permute()
    Permute plusieurs dimensions.
    Exemple : tensor.permute(2, 0, 1)

    .squeeze()
    Supprime les dimensions de taille 1.
    Exemple : tensor.squeeze()

    .unsqueeze()
    Ajoute une dimension de taille 1 à une position spécifique.
    Exemple : tensor.unsqueeze(0)

    .flatten()
    Aplatit le tenseur en 1D.
    Exemple : tensor.flatten()

4. Conversion de type et de device

    .to()
    Change le type de données ou le device (CPU/GPU).
    Exemple : tensor.to(torch.float16) ou tensor.to("cuda")

    .cpu()
    Déplace le tenseur vers le CPU.
    Exemple : tensor.cpu()

    .cuda()
    Déplace le tenseur vers le GPU.
    Exemple : tensor.cuda()

    .float(), .int(), .long()
    Convertit le type de données.
    Exemple : tensor.float()

5. Opérations mathématiques

    .sum(), .mean(), .max(), .min()
    Calcule la somme, moyenne, maximum, ou minimum.
    Exemple : tensor.sum(dim=1)

    .matmul()
    Produit matriciel.
    Exemple : tensor1.matmul(tensor2)

    .abs(), .sqrt(), .exp(), .log()
    Applique des fonctions élémentaires.
    Exemple : tensor.sqrt()

    .clamp()
    Restreint les valeurs entre un intervalle.
    Exemple : tensor.clamp(0, 1)

6. Indexation et découpage

    .gather()
    Rassemble des valeurs selon un axe.
    Exemple : torch.gather(tensor, dim=1, index=indices)

    .masked_select()
    Sélectionne les éléments selon un masque booléen.
    Exemple : tensor.masked_select(mask)

    .index_select()
    Sélectionne des éléments le long d'une dimension.
    Exemple : tensor.index_select(dim=0, index=indices)

7. Gestion de la mémoire

    .clone()
    Crée une copie profonde du tenseur (avec gradients).
    Exemple : tensor.clone()

    .contiguous()
    Force le tenseur à être contigu en mémoire.
    Exemple : tensor.contiguous()

8. Opérations in-place (modifient le tenseur existant)

    _ suffixe
    Les méthodes avec un _ (comme add_(), mul_()) modifient le tenseur directement.
    Exemple : tensor.add_(5).

9. Création de tenseurs

    torch.zeros(), torch.ones(), torch.randn()
    Crée des tenseurs remplis de 0, 1, ou valeurs aléatoires.
    Exemple : torch.zeros(2, 3)

    torch.arange(), torch.linspace()
    Génère des séquences.
    Exemple : torch.arange(0, 10, 2)

10. Autres fonctions utiles

    .numel()
    Retourne le nombre total d'éléments.
    Exemple : tensor.numel()

    .size()
    Retourne la forme du tenseur.
    Exemple : tensor.size()

    .dim()
    Retourne le nombre de dimensions.
    Exemple : tensor.dim()

    torch.cat(), torch.stack()
    Concatène des tenseurs.
    Exemple : torch.cat([tensor1, tensor2], dim=0)
