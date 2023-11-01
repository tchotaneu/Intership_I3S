import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.spatial.distance import squareform, pdist
import numpy as np
import matplotlib.pyplot as plt
from buid_views import DrawCurve as draw


import networkx as nx
import os
from abc import ABC, abstractmethod




class Sae_AI(nn.Module):
    """
      Classe représentant un Auto-encodeur Empilé (Stacked Autoencoder - SAE_Multiview).
      Utilisé pour obtenir une représentation unifiée de l'ensemble de données.
    """
    def __init__(self, input_dim=64, hidden_dims=[1000, 32, 1000], dropout_rate=0.05):
        """
        Initialise le modèle SAE_Multiview

        :param input_dim: Dimension de l'entrée du modèle. Default is 200.
        :param hidden_dims: Liste contenant les dimensions des couches cachées. Default is [200, 30, 200].
        :param dropout_rate: Taux de dropout pour éviter le surajustement. Default is 0.05.
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),  # Première couche de l'encodeur
            nn.ReLU(),  # Activation ReLU
            nn.Dropout(dropout_rate),  # Dropout pour régularisation
            nn.Linear(hidden_dims[0], hidden_dims[1]),  # Deuxième couche de l'encodeur
            nn.ReLU(),  # Activation ReLU
            nn.Dropout(dropout_rate)  # Dropout pour régularisation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),  # Première couche du décodeur
            nn.ReLU(),  # Activation ReLU
            nn.Dropout(dropout_rate),  # Dropout pour régularisation
            nn.Linear(hidden_dims[2], input_dim),  # Deuxième couche du décodeur
            nn.Sigmoid()  # Activation Sigmoid pour obtenir des valeurs entre 0 et 1
        )

    def forward(self, x):
        """
          Passe en avant dans le modèle.

          :param x: Les données d'entrée.
          :return: Les données reconstruites et les données encodées.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    

 

   

    def calculateY_init(self, X_unified,epochs):
        """
        Cette méthode calcule l'embedding initial Y à partir de l'entrée unifiée X.
        
        Args:
        X_unified (np.ndarray): Matrice unifiée, où la première colonne contient les labels et les colonnes suivantes contiennent les caractéristiques.
        epochs: int : le nombre d'epoque d'entrainement du modele 
        Returns:
        np.ndarray: Matrice Y_init_with_labels, où la première colonne contient les labels et les colonnes suivantes contiennent les embeddings calculés.
        
        ### Processus :
        1. **Normalisation des Caractéristiques** :
           Les caractéristiques sont normalisées pour être dans l'intervalle [0, 1].
        
        2. **Entraînement du Modèle** :
           Le modèle est entraîné pour un nombre d'époques spécifié, en utilisant la perte quadratique moyenne (MSE Loss) et l'optimiseur Adam.
        
        3. **Calcul de l'Embedding Initial** :
           Une fois le modèle entraîné, l'embedding initial Y est calculé en utilisant l'encodeur du modèle.
        
        4. **Concaténation des Labels** :
           Les labels sont ensuite concaténés à l'embedding initial pour former la matrice finale Y_init_with_labels.
        """
        # Extraire les labels et les caractéristiques de X_unified
        labels = X_unified[:, 0] 
        features = X_unified[:, 1:]  
        
        # Normaliser les caractéristiques
        X_tensor = torch.FloatTensor(features)   
        X_tensor = (X_tensor - X_tensor.min()) / (X_tensor.max() - X_tensor.min()) 
        
        # Déplacer le modèle et les données vers le GPU si disponible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        X_tensor = X_tensor.to(device)
        
        
        # Définir le critère de perte et l'optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        # Entraîner le modèle
        training_losses = []
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs, _ = self(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
            #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Calculer l'embedding initial Y
        with torch.no_grad():
            self.eval()
            Y_init = self.encoder(torch.tensor(features, dtype=torch.float32).to(device)).cpu().numpy()
        
        # Concaténer les labels à Y_init pour obtenir Y_init_with_labels
        labelsr = np.reshape(labels, (len(labels), 1))
        Y_init_with_labels = np.concatenate((labelsr, Y_init), axis=1)
        
        # Tracer la courbe d'apprentissage et sauvegarder le modèle et Y_init_with_labels
        #Sae_AI.plot_learning_curves(training_losses)

        #torch.save(self.state_dict(), 'sae_model.pth')
        #np.save('Y_init_with_labels.npy', Y_init_with_labels)
        
        return training_losses,Y_init_with_labels

   
    def calculate_Pij(self,views_probabilities,views):
        """
        Calcule la probabilité conjointe p_ij que deux points d'échantillon soient voisins dans toutes les vues.

        :param views_probabilities: List[dict] - Liste de dictionnaires de probabilités pour chaque vue.
        Chaque dictionnaire contient des probabilités symétriques entre les paires de points de données dans une vue.

        :return: dict - Dictionnaire de probabilités conjointes p_ij pour toutes les vues.
        Chaque clé du dictionnaire est une paire de labels de points (i, j), et la valeur correspondante est la probabilité conjointe.

        :Formule mathématique appliquer est l'equation 13 qui est une complementaire al'equation9 :en effet
             1)si  le noeud(j) apparaient dans une vue et que le noeud(i) n'apparait pas dans tous les vues alors p_{ij} =p_{v_{ij}/|V|
             2)si le noeud (i) apparaient dans une seule vue alors le p_{ij} =p_{v_{ij}
             3)dans les autres cas nous appliquons l'equation 9 :  p_{ij} = \frac{\prod_v p_{v_{ij}}}{\prod_v p_{v_{ij}} + \prod_v \sum_{k \neq j} p_{v_{ik}}}

        """
        keys = set()  # Crée un ensemble vide pour stocker toutes les clés

        for view_probs in views_probabilities:
            keys.update(view_probs.keys())  # Ajoute les clés de chaque dictionnaire à l'ensemble

    # Vous avez maintenant un ensemble 'all_keys' contenant toutes les clés uniques de tous les dictionnaires

        num_views = len(views_probabilities)
        Pij = {}  # Dictionnaire pour stocker les probabilités conjointes

        # Parcourir les clés du premier dictionnaire de probabilités (toutes les clés sont les mêmes pour chaque vue)
        # Créez un ensemble pour stocker les clés communes
        common_keys = set(views_probabilities[0].keys())

        # Créez un ensemble pour stocker toutes les clés
        all_keys = set(views_probabilities[0].keys())

        # Parcourez les autres dictionnaires dans views_probabilities
        for view_probs in views_probabilities[1:]:
            # Mettez à jour l'ensemble des clés communes (intersection)
            common_keys.intersection_update(view_probs.keys())
            
            # Mettez à jour l'ensemble de toutes les clés (union)
            all_keys.update(view_probs.keys())
        # Après la boucle
        non_common_keys = all_keys - common_keys  # Obtenez les clés non communes

       # keys = views_probabilities[0].keys()

        for key in  non_common_keys:
            compteur_i=self.count_views(key[0], views)
            compteur_j=self.count_views(key[1],views)
            if ((compteur_j==1) and (compteur_i <num_views)and (compteur_i!=1)):
              #on recherche le couple cle (i,j) dans l'ensemble des view des probapilites
                joint_prob_ij=self.rechearh_valeur(key,views_probabilities)
                Pij[key] = joint_prob_ij/num_views
        
            if compteur_i==1:
               #on recherche le couple cle (i,j) dans l'ensembles des views  des probapilites
                joint_prob_ij=self.rechearh_valeur(key,views_probabilities)
                Pij[key] = joint_prob_ij
                

        for key in common_keys:  
            product_of_probs = 1.0  # Initialisez le produit des probabilités pour cette paire de points (i, j)
            sum_of_other_probs = 0.0  # Initialisez la somme des probabilités pour les autres points (k != j)
            # Parcourir chaque vue et calculez le produit des probabilités p_v_ij et la somme des autres probabilités p_v_ik
            for view_probs in views_probabilities:
                product_of_probs *= view_probs[key]
                # Calculer la somme des autres probabilités
                produit_of_somme = 1.0
                for k in keys:  # Parcourir les labels (i, j)
                    if ((k[1] != key[1]) and ( k[0] == key[0])):  #  nous nous Assurons que k est différent de j et a le même label i
                        sum_of_other_probs += view_probs.get(k, 0.0)  # Utilisons la valeur 0.0 si la clé n'existe pas
                       
                produit_of_somme = produit_of_somme * sum_of_other_probs
            # Calcul de la probabilité conjointe p_ij en utilisant le produit et la somme calculés
            joint_prob_ij = product_of_probs / (product_of_probs + produit_of_somme)

            # Stockez la probabilité conjointe dans le dictionnaire final
            Pij[key] = joint_prob_ij

        return Pij

  
    def count_views(self,label, views):
        """
        Compte le nombre de vues auxquelles un label appartient.

        :param label: Le label à rechercher dans les vues.
        :param views: La liste des vues (tableaux Numpy) où la première colonne contient les labels.
        :return: Le nombre de vues auxquelles le label appartient.
        """
        count = 0  # Initialiser le compteur de vues

        for view in views:  # Parcourir chaque vue dans la liste des vues
            if label in view[:, 0]:  # Si le label est trouvé dans la première colonne de la vue
                count += 1  # Incrémenter le compteur de vues

        return count  # Retourner le compteur de vues
    
    def rechearh_valeur(self,i, setofDictinary):
        for view in setofDictinary:
            if i in view.keys():
                return view[i]
        return None  # Retourner None si la clé n'est pas trouvée dans aucun des dictionnaires de l'ensemble

   ###
    def calcule_qij(self,Y):
        """
        Calcule la matrice des probabilités induites qij à partir de l'embedding Y.
        
        Args:
        Y (np.ndarray): Matrice de l'embedding, où chaque ligne représente un point dans l'espace de dimension réduite.
        
        Returns:
        dict: Dictionnaire où les clés sont des couples de labels et les valeurs sont les probabilités induites qij correspondantes.
        
        ### Objectif
        Nous allons calculer la matrice des distances carrées entre chaque paire de points dans l'embedding \( Y \). 
        La matrice résultante, \( D \), aura la distance carrée entre le noeud de position \( i \) et le noeud de position \( j \) 
        à la position \( D[i, j] \).
        
        ### Détails du Code
        1. **Calcul de `sum_Y`** :
          ```python
          sum_Y = np.sum(np.square(Y), 1)
          ```
          Ici, pour chaque point dans \( Y \), on calcule la somme des carrés de ses coordonnées. 
          Le résultat, `sum_Y`, est un vecteur où chaque élément correspond à la somme des carrés des coordonnées d'un noeud dans \( Y \).
        
        2. **Calcul de la matrice des distances carrées, `D`** :
          ```python
          D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
          ```
          Pour calculer la matrice des distances carrées entre chaque paire de noeud, on utilise la formule de distance euclidienne carrée 
          entre deux noeuds \( x \) et \( y \) :
          \[ d(x, y)^2 = (x_1 - y_1)^2 + (x_2 - y_2)^2 + \ldots + (x_n - y_n)^2 \]
          \[ d(x, y)^2 = \sum_{k=1}^{n} (x_k - y_k)^2 \]
          \[ d(x, y)^2 = \sum_{k=1}^{n} x_k^2 - 2x_ky_k + y_k^2 \]
          \[ d(x, y)^2 = \sum_{k=1}^{n} x_k^2 + \sum_{k=1}^{n} y_k^2 - 2 \sum_{k=1}^{n} x_ky_k \]
          
          En utilisant cette formule, la matrice des distances carrées, \( D \), est calculée comme suit :
          - \( \text{np.dot}(Y, Y.T) \) calcule le produit scalaire entre chaque paire de noeud, ce qui correspond à \( 2 \sum_{k=1}^{n} x_ky_k \).
          - \( -2 * \text{np.dot}(Y, Y.T) \) inverse le signe et double le produit scalaire.
          - \( \text{np.add}(-2 * \text{np.dot}(Y, Y.T), \text{sum_Y}) \) ajoute la somme des carrés de chaque noeud à chaque ligne de la matrice, 
            ce qui correspond à ajouter \( \sum_{k=1}^{n} x_k^2 \) à chaque élément de la ligne.
          - Enfin, \( \text{np.add}(\ldots.T, \text{sum_Y}) \) ajoute la somme des carrés de chaque noeud à chaque colonne de la matrice, 
            ce qui correspond à ajouter \( \sum_{k=1}^{n} y_k^2 \) à chaque élément de la colonne.
        
        ### Résultat
        La matrice \( D \) contient les distances carrées entre chaque paire de points dans l'embedding \( Y \). 
        Chaque élément \( D[i, j] \) représente la distance carrée entre le point \( i \) et le point \( j \) dans l'embedding.
        """
        Y_labels = Y[:, 0].astype(int)  # Extraire la première colonne contenant les labels
        Y_features = Y[:, 1:]  # Exclure la première colonne contenant les labels
        
        n = Y_features.shape[0]  # Nombre de points dans l'embedding
        Qij = {}  # Initialisation du dictionnaire Q
        
        # Calcul de la matrice des distances carrées
        sum_Y = np.sum(np.square(Y_features), 1)
        D = np.add(np.add(-2 * np.dot(Y_features, Y_features.T), sum_Y).T, sum_Y)
        
        # Calcul de la matrice des probabilités induites qij
        num = 1 / (1 + D)  # Numérateur de qij
        np.fill_diagonal(num, 0)  # Remplir la diagonale avec des zéros pour éviter la division par zéro
        
        # Remplissons du dictionnaire Qijavec les couples de labels comme clés et les probabilités induites comme valeurs
        for i in range(n):
            for j in range(n):
                if i != j:  # Exclure la diagonale
                    key = (Y_labels[i], Y_labels[j])  # Créer une clé à partir des labels de i et j
                    Qij[key] = num[i, j] / np.sum(num)  # Normalisation et ajout au dictionnaire
        
        return Qij

   
    def optimisation_divergence_kl(self,Y1, P_ij, Q_ij, taux_apprentissage=0.01, nombre_iterations=10):
        """
        Optimise Y en minimisant la divergence KL entre P et Q.
        Optimise Y en minimisant la divergence KL entre P et Q.
        La fonction de coût C est définie comme la divergence de Kullback-Leibler (KL) entre P et Q :
        C = KL(P||Q) = ∑_i ∑_j p_{ij} log(p_{ij} / q_{ij})
        Le gradient de la fonction de coût C par rapport à y_i est calculé comme suit :
        ∂C/∂y_i = 4 ∑_j (p_{ij} - q_{ij}) (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
      
        :param Y: Représentation initiale en basse dimension (tableau NumPy de forme (n_echantillons, n_caracteristiques))
        :param P: Dictionnaire représentant les probabilités en haute dimension avec les clés (i, j)
        :param Q: Dictionnaire représentant les probabilités en basse dimension avec les clés (i, j)
        :param taux_apprentissage: Taux d'apprentissage pour l'optimisation par descente de gradient
        :param nombre_iterations: Nombre d'itérations pour l'optimisation
        :return: Représentation optimisée en basse dimension Y
        """
        labels=Y1[:,0]
        Y=Y1[:,1:]
        norme_du_gradient_history = []
        for iteration in range(nombre_iterations):
            # Initialiser le gradient
            grad_Y = np.zeros_like(Y)
            
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    if i != j:
                        # Calculer le gradient pour chaque paire (i, j)
                        pij = P_ij.get((i, j), 0)
                        qij = Q_ij.get((i, j), 0)
                        grad_Y[i] += 4 * (pij - qij) * (1 + np.linalg.norm(Y[i] - Y[j]) ** 2) ** -1 * (Y[i] - Y[j])
            
            # Mettre à jour Y en utilisant le gradient
            Y -= taux_apprentissage * grad_Y
            norme_du_gradient = np.linalg.norm(grad_Y)
            norme_du_gradient_history.append(norme_du_gradient)
            # Optionnellement, imprimer la progression
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Norme du gradient {np.linalg.norm(grad_Y)}")
        #labelsr = np.reshape(labels, (len(labels), 1))
        #embedding = np.concatenate((labelsr, Y), axis=1)
        
        labelsr= labels.reshape(-1, 1)
        embedding= np.hstack((labelsr,Y))
       
        return norme_du_gradient_history,embedding


# reference:
# https://github.com/sezinata/MANE.git

class ManeAI(nn.Module):  
    
    def __init__(self,nviews,dimensions,device, len_common_nodes, embed_freq, batch_size, negative_sampling_size=10):
        super(ManeAI, self).__init__()
        self.n_embedding = len_common_nodes
        self.embed_freq = embed_freq
        self.num_net = nviews
        self.negative_sampling_size = negative_sampling_size
        self.node_embeddings = nn.ModuleList()
        self.neigh_embeddings = nn.ModuleList()
        self.embedding_dim = dimensions
        self.device = device
        self.batch_size = batch_size
        for n_net in range(self.num_net):  # len(G)
            self.node_embeddings.append(nn.Embedding(len_common_nodes, self.embedding_dim))
            self.neigh_embeddings.append(nn.Embedding(len_common_nodes, self.embedding_dim))

       

    def forward(self, count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, hyp1, hyp2):
        cost1 = [nn.functional.logsigmoid (  torch.bmm(self.neigh_embeddings[i](Variable(torch.LongTensor(
            neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
            self.embedding_dim), self.node_embeddings[i](Variable(
            torch.LongTensor(nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                self.device))).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(
            2))).squeeze().mean() + nn.functional.logsigmoid(torch.bmm(self.neigh_embeddings[i](
            self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.neigh_embeddings[i](Variable(
                    torch.LongTensor(
                        neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                            self.embedding_dim).size(1) * self.negative_sampling_size, replacement=True).to(
                self.device)).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
            self.embedding_dim).neg(), self.node_embeddings[i](Variable(
            torch.LongTensor(nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                self.device))).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(0) for
                 i in range(self.num_net)]

        # First order collaboration
        cost2 = [[hyp1 * (nn.functional.logsigmoid(torch.bmm(self.node_embeddings[j](Variable(torch.LongTensor(
            nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim),
            self.node_embeddings[i](Variable(torch.LongTensor(
                nodes_idx_nets[i][shuffle_indices_nets[i][
                                  count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][
                    count:count + self.batch_size]), -1).unsqueeze(
                2))).squeeze().mean() + nn.functional.logsigmoid(
            torch.bmm(self.node_embeddings[j](self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.node_embeddings[j](Variable(
                    torch.LongTensor(
                        nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim).size(
                    1) * self.negative_sampling_size,
                replacement=True).to(self.device)).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                                                        self.embedding_dim).neg(), self.node_embeddings[i](Variable(
                torch.LongTensor(
                    nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(
            0))
                  for i in range(self.num_net) if i != j] for j in range(self.num_net)]

        # Second order collaboration

        cost3 = [[hyp2 * (nn.functional.logsigmoid(torch.bmm(self.neigh_embeddings[j](Variable(torch.LongTensor(
            neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim),
            self.node_embeddings[i](Variable(torch.LongTensor(
                nodes_idx_nets[i][shuffle_indices_nets[i][
                                  count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][
                    count:count + self.batch_size]), -1).unsqueeze(
                2))).squeeze().mean() + nn.functional.logsigmoid(
            torch.bmm(self.neigh_embeddings[j](self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.neigh_embeddings[j](Variable(
                    torch.LongTensor(
                        neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim).size(
                    1) * self.negative_sampling_size,
                replacement=True).to(self.device)).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                                                        self.embedding_dim).neg(), self.node_embeddings[i](Variable(
                torch.LongTensor(
                    nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(
            0))
                  for i in range(self.num_net) if i != j] for j in range(self.num_net)]

        sum_cost2 = []
        [[sum_cost2.append(j) for j in i] for i in cost2]

        sum_cost3 = []
        [[sum_cost3.append(j) for j in i] for i in cost3]

        return -(torch.mean(torch.stack(cost1)) + sum(sum_cost2) / len(sum_cost2) + sum(sum_cost3) / len(sum_cost3)) / 3
    
