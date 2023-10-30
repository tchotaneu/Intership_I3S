import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from abc import ABC, abstractmethod




class Sae_AI(nn.Module):
    """
      Classe représentant un Auto-encodeur Empilé (Stacked Autoencoder - SAE_Multiview).
      Utilisé pour obtenir une représentation unifiée de l'ensemble de données.
    """
    def __init__(self, input_dim=64, hidden_dims=[200, 32, 200], dropout_rate=0.05):
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

    @staticmethod
    def generate_unified_X(views):
        """
        Génère un ensemble de données unifié en concaténant différentes vues.

        :param views: Liste des différentes vues à concaténer.
        :return: Ensemble de données unifié.
        """
        if not all(isinstance(view, np.ndarray) for view in views):
            raise ValueError("Toutes les vues doivent être des tableaux NumPy.")

        all_labels = set().union(*[set(view[:, 0]) for view in views])
        total_features = sum(view.shape[1] - 1 for view in views) + 1
        X_unified = np.zeros((len(all_labels), total_features))
        X_unified[:, 0] = list(all_labels)

        start_idx = 1
        for view in views:
            for row in view:
                label = row[0]
                features = row[1:]
                idx = np.where(X_unified[:, 0] == label)[0][0]
                X_unified[idx, start_idx:start_idx + len(features)] = features
            start_idx += view.shape[1] - 1

        return X_unified

    @staticmethod
    def plot_learning_curves(training_losses, validation_losses=None):
        """
        Affiche les courbes d'apprentissage du modèle.

        :param training_losses: Liste des pertes d'entraînement à chaque époque.
        :param validation_losses: Liste des pertes de validation à chaque époque (optionnel).
        """
        plt.plot(range(len(training_losses)), training_losses, label='Training Loss')
        if validation_losses is not None:
            plt.plot(range(len(validation_losses)), validation_losses, label='Validation Loss', linestyle='dashed')
        plt.xlabel('Epoques')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()
        

    def generate_node2vec_embeddings(self,graph_paths, node2vec_params_list):
        """
          Génère des embeddings Node2Vec pour différentes vues.

          :param graph_paths: Liste des chemins vers les fichiers de graphes.
          :param node2vec_params_list: Liste des paramètres Node2Vec pour chaque graphe.
          :return: Liste des embeddings pour chaque vue.
        """
        views = []
        for graph_path, node2vec_params in zip(graph_paths, node2vec_params_list):
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Le fichier {graph_path} n'existe pas.")
            try:
                G = nx.read_edgelist(graph_path)
            except Exception as e:
                raise ValueError(f"Impossible de lire le graphe à partir du fichier {graph_path}. Erreur : {str(e)}")
            node2vec = Node2Vec(G, dimensions=node2vec_params['dimensions'], walk_length=node2vec_params['walk_length'],
                                num_walks=node2vec_params['num_walks'], workers=node2vec_params['workers'])
            model = node2vec.fit(window=node2vec_params['window'], min_count=node2vec_params['min_count'],
                                batch_words=node2vec_params['batch_words'], sg=1)
            embeddings = np.array([[int(node)] + list(model.wv[str(node)]) for node in G.nodes()])
            views.append(embeddings)
        return views

    @staticmethod
    def calculate_P_v_ij(views, k):
          """
          Calcule la probabilité symétrique p_v_ij pour chaque point d'échantillon dans chaque vue.

          La probabilité symétrique est calculée selon la fusion des équations 6 et 8 :
          p_v_ij = exp(-||x_{v_i} - x_{v_j}||^2_2 / (2 * sigma_{v_i}^2)) / sum(exp(-||x_{v_i} - x_{v_k}||^2_2 / (2 * sigma_{v_i}^2)))

          :param views: List[np.ndarray] - Liste des vues. Chaque vue est une matrice numpy bidimensionnelle avec les labels dans la première colonne et les caractéristiques dans les colonnes suivantes.
          :param k: int - Nombre effectif de voisins les plus proches à considérer pour le calcul de sigma.
          :return: List[dict] - Liste de dictionnaires où, pour chaque vue, chaque clé est un couple de labels et la valeur est la probabilité symétrique correspondante.
          """
          result_dicts = []
          epsilon = 1e-10  # Petite valeur ajoutée au dénominateur pour éviter la division par zéro

          for view in views:
              result_dict = {}
              labels = view[:, 0].astype(int)  # Extraire les labels de la première colonne et les convertir en entiers
              features = view[:, 1:]  # Extraire les caractéristiques des colonnes suivantes

              # Calculer les distances euclidiennes au carré entre tous les points de la vue
              squared_distances = squareform(pdist(features, 'sqeuclidean'))

              # Calculer sigma pour chaque point
              sigmas_squared = SAE_Multiview.calculate_sigmas_squared(squared_distances, k)  # Assurez-vous que cette fonction renvoie sigma^2

              # Calculer la probabilité symétrique p_v_ij pour chaque paire de points dans la vue
              num_points = view.shape[0]
              for i in range(num_points):
                  for j in range(num_points):
                      if i != j:  # Exclure la diagonale, car p_v_ii = 0
                          key = (labels[i], labels[j])  # Créer une clé à partir des labels des points i et j
                          distance = squared_distances[i, j]
                          sigma_squared = sigmas_squared[i]
                          p_v_ij = np.exp(-distance / (2 * sigma_squared + epsilon))
                          result_dict[key] = p_v_ij  # Ajouter la probabilité symétrique au dictionnaire avec la clé correspondante

              # Normaliser les probabilités dans le dictionnaire
              total_probability = sum(result_dict.values())
              for key in result_dict:
                  result_dict[key] /= total_probability

              result_dicts.append(result_dict)  # Ajouter le dictionnaire de probabilités symétriques de cette vue à la liste

          return result_dicts

    @staticmethod
    def calculate_sigmas_squared(squared_distances, k):
        """
        Calcule sigma^2 pour chaque point dans une vue, de manière à ce que l'entropie de la distribution sur les voisins soit égale à log(k).

        :param squared_distances: np.ndarray - Matrice des distances euclidiennes au carré entre tous les points de la vue.
        :param k: int - Nombre effectif de voisins les plus proches à considérer pour le calcul de l'entropie.
        :return: np.ndarray - Vecteur de sigma^2 pour chaque point dans la vue.
        """
        num_points = squared_distances.shape[0]
        sigmas_squared = np.zeros(num_points)

        def entropy(sigma_squared, distances):
            p_j = np.exp(-distances / (2 * sigma_squared))
            p_j /= p_j.sum()  # Normaliser les probabilités
            return -np.sum(p_j * np.log(p_j + 1e-10))  # Calculer l'entropie

        for i in range(num_points):
            distances = squared_distances[i, :]
            target_entropy = np.log(k)

            # Utiliser la recherche par dichotomie pour trouver sigma^2
            sigmas_squared[i] = bisect(
                lambda sigma_squared: entropy(sigma_squared, distances) - target_entropy,
                a=1e-10,  # borne inférieure de sigma^2
                b=1e10   # borne supérieure de sigma^2
            )

        return sigmas_squared

    @staticmethod
    def calculate_Pij(views_probabilities):
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

        num_views = len(views_probabilities)
        Pij = {}  # Dictionnaire pour stocker les probabilités conjointes

        # Parcourir les clés du premier dictionnaire de probabilités (toutes les clés sont les mêmes pour chaque vue)
        keys = views_probabilities[0].keys()

        for key in keys:
            #compteur_i=SAE_Multiview.count_views(key[0], views)
            #compteur_j=SAE_Multiview.count_views(key[1],views)
            #if ((compteur_j==1) and (compteur_i <num_views)and (compteur_i!=1)):
              #on recherche le couple cle (i,j) dans l'ensemble des view des probapilites
             #   joint_prob_ij=SAE_Multiview.rechearh_valeur(key,views_probabilities)
             #   Pij[key] = joint_prob_ij/num_views
             #   continue
            #if compteur_i==1:
               #on recherche le couple cle (i,j) dans l'ensembles des view des probapilites
               # joint_prob_ij=SAE_Multiview.rechearh_valeur(key,views_probabilities)
              #  Pij[key] = joint_prob_ij
             #   continue
            product_of_probs = 1.0  # Initialisez le produit des probabilités pour cette paire de points (i, j)
            sum_of_other_probs = 0.0  # Initialisez la somme des probabilités pour les autres points (k != j)
            print("etape1 fait")
            # Parcourir chaque vue et calculez le produit des probabilités p_v_ij et la somme des autres probabilités p_v_ik
            for view_probs in views_probabilities:
                product_of_probs *= view_probs[key]
                print("etape2 fait")
                # Calculer la somme des autres probabilités
                produit_of_somme = 1.0
                for k in keys:  # Parcourir les labels (i, j)
                    if ((k[1] != key[1]) and ( k[0] == key[0])):  #  nous nous Assurons que k est différent de j et a le même label i
                        sum_of_other_probs += view_probs.get(k, 0.0)  # Utilisez 0.0 si la clé n'existe pas
                        print(sum_of_other_probs)
               # print("etape3 fait")

                produit_of_somme = produit_of_somme * sum_of_other_probs
            print(len(keys)-1)
            # Calcul de la probabilité conjointe p_ij en utilisant le produit et la somme calculés
            joint_prob_ij = product_of_probs / (product_of_probs + produit_of_somme)

            # Stockez la probabilité conjointe dans le dictionnaire final
            Pij[key] = joint_prob_ij

        return Pij

    @staticmethod
    def count_views(label, views):
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
    @staticmethod
    def rechearh_valeur(i, setofDictinary):
        for view in setofDictinary:
            if i in view.keys():
                return view[i]
        return None  # Retourner None si la clé n'est pas trouvée dans aucun des dictionnaires de l'ensemble



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
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Calculer l'embedding initial Y
        with torch.no_grad():
            self.eval()
            Y_init = self.encoder(torch.tensor(features, dtype=torch.float32).to(device)).cpu().numpy()
        
        # Concaténer les labels à Y_init pour obtenir Y_init_with_labels
        labelsr = np.reshape(labels, (len(labels), 1))
        Y_init_with_labels = np.concatenate((labelsr, Y_init), axis=1)
        
        # Tracer la courbe d'apprentissage et sauvegarder le modèle et Y_init_with_labels
        Sae_AI.plot_learning_curves(training_losses)
        torch.save(self.state_dict(), 'sae_model.pth')
        np.save('Y_init_with_labels.npy', Y_init_with_labels)
        
        return Y_init_with_labels

    @staticmethod
    def calcule_qij(Y):
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
        
        # Remplissage du dictionnaire Qijavec les couples de labels comme clés et les probabilités induites comme valeurs
        for i in range(n):
            for j in range(n):
                if i != j:  # Exclure la diagonale
                    key = (Y_labels[i], Y_labels[j])  # Créer une clé à partir des labels de i et j
                    Qij[key] = num[i, j] / np.sum(num)  # Normalisation et ajout au dictionnaire
        
        return Qij

    @staticmethod
    def optimisation_divergence_kl(Y, P, Q, taux_apprentissage=0.01, nombre_iterations=1000):
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
        for iteration in range(nombre_iterations):
            # Initialiser le gradient
            grad_Y = np.zeros_like(Y)
            
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    if i != j:
                        # Calculer le gradient pour chaque paire (i, j)
                        pij = P.get((i, j), 0)
                        qij = Q.get((i, j), 0)
                        grad_Y[i] += 4 * (pij - qij) * (1 + np.linalg.norm(Y[i] - Y[j]) ** 2) ** -1 * (Y[i] - Y[j])
            
            # Mettre à jour Y en utilisant le gradient
            Y -= taux_apprentissage * grad_Y
            
            # Optionnellement, imprimer la progression
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Norme du gradient {np.linalg.norm(grad_Y)}")
        
        return Y


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
    
