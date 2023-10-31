#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
+-------------------------------------------------------------------------------------+
| This file is part of AMINE                                                          |
|                                                                                     |
| AMINE is free software: you can redistribute it and/or modify it under the terms of |
| the GNU General Public License as published by the Free Software Foundation, either |
| version 3 of the License, or (at your option) any later version.                    |
| You should have received a copy of the GNU General Public License along with AMINE. |
| If not, see <http://www.gnu.org/licenses/>.                                         |
|                                                                                     |
| Author: Claude Pasquier (I3S Laboratory, CNRS, Université Côte d'Azur)              |
| Contact: claude.pasquier@univ-cotedazur.fr                                          |
| Created on decembre 20, 2022                                                        |
+-------------------------------------------------------------------------------------+

Various graph models.

Each class defined here is instanciated with a networkx graph
After instanciation, it is possible to get, according to the model,
the closest nodes to a specific node with the method 'get_most_similar'
"""
import sys # sys.exit() 
from datasets import Datasets
import pathlib
from abc import ABC, abstractmethod
from typing import Union, Iterable
import model_NN 
import networkx as nx
import numpy as np
from scipy.optimize import bisect
from gensim.models import FastText, Word2Vec
from scipy.spatial import distance
#import matplotlib.pyplot as plt
import os
from dimension_reduction import node2vec
from dimension_reduction.pecanpy import node2vec as n2v
import buid_views
import torch
#import torch.nn as nn
#import torch.optim as optim
from torch.autograd import Variable
import time
import random
import gc
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import bisect
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
#from node2vec import Node2Vec # appelle de node2vec de mane 

class Model(ABC):
    """
    abstract class.

    Three implementation are proposed:
        - Node2vec,
        - RandomWalk
        - SVD.
    """

    @abstractmethod
    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes ; method implemented by each subclass.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """


class Node2vec(Model):
    """Node2vec model."""

    def __init__(self):
        """Declare variables."""
        self.model = None
        self.num_walks = 20  # 10
        self.walk_length = 100  # 80
        self.directed = False
        self.param_p = 1  # 4  # 0.15
        self.param_q = 1  # 2
        self.dimensions = 64  # 128
        self.window_size = 5  # 10
        self.workers = 4
        self.epoch = 10  # 10
        
    def init(self, G: nx.Graph, list_nodes: Iterable = None, precomputed: str = None):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G           : nx.Graph
                      the graph used to initialize the model
        list_nodes  : Iterable, optional
                      specify an order of the nodes to be used, default is None
        precomputed : str or file-like object, optional
                      None or path to the precomputed model that must be used, default is None

        """
        if list_nodes is None:
            list_nodes = list(G.nodes)
        if precomputed and pathlib.Path(precomputed).is_file():
            self.model = Word2Vec.load(precomputed)
        else:
            for node in G.nodes():
                for nbr in sorted(G.neighbors(node)):
                    G[node][nbr]["weight"] = 1 - abs(
                        G.nodes[node]["weight"] - G.nodes[nbr]["weight"]
                    )
            self.compute_embedding(G, list_nodes)
            if precomputed:
                self.save(precomputed)

    def compute_embedding(self, G: nx.Graph, list_nodes: list):
        """
        Compute embedding.

        Parameters
        ----------
        G           : nx.Graph
                      the processed graph
        list_nodes  : list of nodes
                      the list of start nodes from the randomwalk

        """
        use_pecanpy = False
        if use_pecanpy:
            # from pecanpy import node2vec as n2v
            graph = n2v.SparseOTF(
                p=self.param_p,
                q=self.param_q,
                workers=self.workers,
                verbose=False,
                extend=True,
            )
            A = np.array(
                nx.adjacency_matrix(
                    G, nodelist=sorted(G.nodes), weight="weight"
                ).todense(),
                dtype=np.float_,
            )
            # isolated_nodes = np.where(~A.any(axis=1))[0]
            # print(np.where(~A.any(axis=0))[0])
            # print(nx.is_connected(G))
            # A = np.delete(A, isolated_nodes, axis=0)
            # A = np.delete(A, isolated_nodes, axis=1)
            graph.from_mat(A, sorted(G.nodes))
            walks = graph.simulate_walks(
                num_walks=self.num_walks,
                walk_length=self.walk_length,
                list_nodes=list_nodes,
            )
        else:
            graph = node2vec.Graph(G, self.directed, self.param_p, self.param_q)
            graph.preprocess_transition_probs()
            walks = graph.simulate_walks(
                self.num_walks, self.walk_length, nodes=list_nodes
            )

        # Learn embeddings by optimizing the Skipgram objective using SGD.
        walks = [list(map(str, walk)) for walk in walks]
        # import pickle
        # with open("/home/cpasquie/Téléchargements/test.txt", "wb") as fp:   #Pickling
        #     pickle.dump(walks, fp)
        # dd
        # with open("/home/cpasquie/Téléchargements/test.txt", "rb") as fp:   # Unpickling
        #     walks = pickle.load(fp)
        use_fasttext = False
        if use_fasttext:
            self.model = FastText(
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=1,
                sentences=walks,
                epochs=self.epoch,
                max_n=0,
                sg=1,
            )
        else:
            self.model = Word2Vec(
                walks,
                vector_size=self.dimensions,  # size=self.dimensions,
                window=self.window_size,
                min_count=5,
                negative=5,
                sg=1,
                workers=self.workers,
                epochs=self.epoch,
            )  # iter=self.epoch)
            # bout de code ajouter pour sauvarder l'embedding 
            words = self.model.wv.index_to_key
            vectors = self.model.wv.vectors
            embedding = np.column_stack((words, vectors))
            fichier="sauvegarde/"+self.directory+"/embedding/embedding.txt"
            np.savetxt(fichier, embedding, delimiter=' ', fmt='%f')


    def save(self, fname_or_handle: str):
        """
            Save the model to file.

        Parameters
        ----------
        fname_or_handle : str or file-like object
                          path or handle to file where the model will be persisted

        """
        self.model.save(fname_or_handle)

    def load(self, fname_or_handle: str):
        """
        Load a previously saved model from a file.

        Parameters
        ----------
        fname_or_handle : str or file-like object
                          path or handle to file that contains the model

        """
        self.model = Word2Vec.load(fname_or_handle)

    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        return [int(x[0]) for x in self.model.wv.similar_by_word(str(elt), topn=number)]

    def get_distance(self, elt1: str, elt2: str):
        """
        Return the distance between two elements.

        Parameters
        ----------
        elt1 : str
            first element
        elt2 : str
            second element

        """
        return self.model.wv.distance(str(elt1), str(elt2))

    def get_vector(self, elt: Union[str, int]):
        """
        Get the vector encoding the element

        Parameters
        ----------
        elt : Union[str, int]
            the element

        Returns
        -------
        vector
            the vector encoding the element
        """
        return self.model.wv.get_vector(str(elt))


class RandomWalk(Model):
    """
    RandomWalk model.
    """

    # convergence criterion - when vector L1 norm drops below 10^(-6)
    # (this is the same as the original RWR paper)
    conv_threshold = 0.000001

    def __init__(self):
        """Declare variables."""
        self.nodelist = None
        self.walk_length = 0
        self.restart_prob = 0
        self.T = None

    def init(self, G: nx.Graph):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G   : nx.Graph
              the graph used to initialize the model

        """
        self.nodelist = sorted(G.nodes)
        self.walk_length = 200
        self.restart_prob = 0.7

        for node in G.nodes():
            for nbr in sorted(G.neighbors(node)):
                G[node][nbr]["weight"] = 1 - abs(
                    G.nodes[node]["weight"] - G.nodes[nbr]["weight"]
                )

        # Create the adjacency matrix of G
        A = np.array(
            nx.adjacency_matrix(G, nodelist=self.nodelist, weight="weight").todense(),
            dtype=np.float_,
        )
        # Create the degree matrix
        D = np.diag(np.sum(A, axis=0))

        # The Laplacian matrix L, not used here is equal to D - A

        # Compute the inverse of D
        # Several solutions are possible
        #     - first solution: numpy.inverse
        #       inverse_of_d = numpy.linalg.inv(D)
        #     - second solution: numpy.solve
        #       inverse_of_d = numpy.linalg.solve(D, numpy.identity(len(nodes_list))
        #     - third solution, as the matrix is diagonal, one can use
        #       the inverse of the diagonal values
        #       inverse_of_d = np.diag(1 / np.diag(D))

        inverse_of_d = np.diag(1 / np.diag(D))

        # compute the transition matrix
        self.T = np.dot(inverse_of_d, A)

    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        arr = [0] * len(self.nodelist)
        arr[elt] = 1
        p_0 = np.array(arr)
        state_matrix = p_0
        for _ in range(self.walk_length):

            # evaluate the next state vector
            p_1 = (1 - self.restart_prob) * np.dot(
                state_matrix, self.T
            ) + self.restart_prob * p_0

            # calculate L1 norm of difference between p^(t + 1) and p^(t),
            # for checking the convergence condition
            diff_norm = np.linalg.norm(np.subtract(p_1, state_matrix), 1)
            if diff_norm < RandomWalk.conv_threshold:
                break
        state_matrix = p_1
        result = sorted(
            enumerate(state_matrix.tolist()), key=lambda res: res[1], reverse=True
        )
        return [int(x[0]) for x in result][1 : number + 1]


class SVD(Model):
    """SVD model."""

    def __init__(self):
        """Declare variables."""
        self.nodelist = None
        self.most_similar = []

    def init(self, G: nx.Graph):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G   : nx.Graph
              the graph used to initialize the model

        """
        self.nodelist = sorted(G.nodes)
        A = np.array(
            nx.adjacency_matrix(G, sorted(G.nodes), weight=None).todense(),
            dtype=np.float_,
        )
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        reduced_dimension = A.shape[0] // 5
        reduced_matrix = U * S
        reduced_matrix = reduced_matrix[:, 0:reduced_dimension]
        self.most_similar = []
        for ctr in range(reduced_matrix.shape[0]):
            dist = distance.cdist(
                reduced_matrix[ctr : ctr + 1], reduced_matrix[0:], "cosine"
            )
            self.most_similar.append(
                [
                    x[0]
                    for x in sorted(
                        list(enumerate(dist[0].tolist())), key=lambda x: x[1]
                    )
                ][1:]
            )

    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        return self.most_similar[elt][:number]
    
class MultiView(Model):

    def __init__(self):
        """Declare variables."""
       
        self.singleton=True
        self.G = nx.Graph()
        self.G1 =nx.Graph()
        self.directory = None
        self.model = None
        self.device = None
        self.cuda=True
        self.read_pair=False # True si nous voulons utiliser les fichiers de la derniers operations 
        self.output= True # true si nous voulons sauvegarde les fichiers 
        self.nviews=2
        self.similarite_embedding =None
        self.metrique=None
        self.choice=None
        self.builGrap=buid_views.GraphBuilder() 
        self.drawCurve=buid_views.DrawCurve() 
        self.choiceOfConstructGraph= {
            '1': self.builGrap.construct_graph1,
            '2': self.builGrap.construct_graph2,
            '3': self.builGrap.construct_graph3,
            '4': self.builGrap.construct_graph4,
            '5': self.builGrap.construct_graph5,
            '6': self.builGrap.construct_graph6,
            '7': self.builGrap.construct_graph7,
        }
        self.choice_metrique= {
            '1': "cosinus",
            '2': "pearson",
            '3': "eucludian",
            '4': "produitsalaire"
        }
        self.parametreNode2vec=[ {'p':0.25,  'q':0.5, 'window_size':10, 'num_walks': 10, 'walk_length': 20, },
                                 {'p':1.5,  'q':0.5, 'window_size':10, 'num_walks': 20, 'walk_length': 25, },
                                ]  
        
    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        label=float(elt)
        indice_label = np.where(self.model[:,0]== label)[0][0]
        labels_vectors=self.model[:, 0]
        reference_vector = self.model[indice_label, 1:]
        vectors = self.model[:, 1:]
        # choix de la metrique 
        if self.metrique=="cosinus":
            indices_similaires = self.find_similar_cosine(vectors , reference_vector)
        elif self.metrique=="pearson":
            indices_similaires=self.find_similar_pearson(vectors , reference_vector)
        elif self.metrique=="eucludian":
            indices_similaires=self.find_similar_euclidean(vectors , reference_vector)
        else :
            print("metrique non defini ")
        labels_similaires = labels_vectors [indices_similaires]
        
        return list(map(int, labels_similaires.tolist()))
    
    def find_similar_cosine(self, vectors, reference_vector):
        
        # Calculer la similarité cosinus une seule fois entre le vecteur de référence et tous les vecteurs
        similarities = cosine_similarity([reference_vector], vectors).flatten()
        #normalized_embeddings= self.model[:, 1:] / np.linalg.norm(self.model[:, 1:], axis=1, keepdims=True)

        # Trier les indices des vecteurs en fonction de leur similarité cosinus (en ordre décroissant)
        similar_indices = np.argsort(similarities)[::-1]

        return similar_indices

    def find_similar_pearson(self, vectors, reference_vector):
        """
        
        """

        # Calculer la corrélation une seule fois entre le vecteur de référence et tous les vecteurs
        correlations = np.array([pearsonr(reference_vector, vector)[0] for vector in vectors])
        # Trier les indices des vecteurs en fonction de leur corrélation (en ordre décroissant)
        similar_indices  = np.argsort(correlations)[::-1]

        return  similar_indices

    def find_similar_euclidean(self, vectors, reference_vector):
    
        # Calculer la distance euclidienne une seule fois entre le vecteur de référence et tous les vecteurs
        distances = np.linalg.norm(vectors - reference_vector, axis=1)

        # Trier les indices des vecteurs en fonction de leur distance euclidienne (en ordre croissant)
        similar_indices = np.argsort(distances)

        return similar_indices

    
    def find_similar_pscalaire( self,elt: str, number: int):

    
        """
        Renvoie les étiquettes des top_n vecteurs qui ont le produit scalaire le plus élevé avec le vecteur associé à l'étiquette de référence.

        Args:
        - data (np.array): Tableau où la première colonne contient des étiquettes et les colonnes suivantes contiennent les coordonnées des vecteurs.
        - elt (str): Étiquette du vecteur de référence.
        - top_n (int): Nombre d'étiquettes à renvoyer.

        Returns:
        - list of str: Étiquettes des top_n vecteurs.
        """
        # Trouver le vecteur associé à l'étiquette de référence
        lab=float(elt)
        #top_n=10
        data=self.model
        reference_vector = data[data[:, 0] == lab][0, 1:]

        # Séparation des étiquettes et des vecteurs
        labels = data[:, 0]
        vectors = data[:, 1:]

        # Calcul des produits scalaires
        dot_products = np.dot(vectors, reference_vector)

        # Tri des étiquettes en fonction des produits scalaires et sélection des top_n
        sorted_indices = np.argsort(dot_products)[::-1]  # Tri décroissant
        top_labels = list(map(int,labels[sorted_indices].tolist() )) 

        return top_labels


   
    def init(self, G: nx.Graph,
                  p_valeur:float,
                  choice:None,
                  choice_metrique:None,
                  dossier:None,
                  no_singleton:None,
                  my_dict:None):
        self.metrique=choice_metrique
        self.directory=dossier
        valuetrue = my_dict[1]
        list_noeuds_isole=self.builGrap.isolated_low_nodes(G)
        print("nombre des noeuds de P_valeurs (0.05) isolée", len(list_noeuds_isole), list_noeuds_isole & valuetrue,len(list_noeuds_isole & valuetrue))
        list_noeuds_totale=self.builGrap.low_nodes(G)
        print("nombre des noeuds de P_valeurs (0.05)  total dans le graphe  ", len(list_noeuds_totale),list_noeuds_totale & valuetrue,len(list_noeuds_totale & valuetrue))
        liste_des_noeudsconnecte005=self.builGrap.get_low_pvalue_nodes_with_low_pvalue_neighbors(G)
        print("nombre des noeuds de P_valeurs (0.05) connectée dans le graphe ", len(liste_des_noeudsconnecte005),set(liste_des_noeudsconnecte005) & valuetrue, len(set(liste_des_noeudsconnecte005) & valuetrue))
        list_noeuds_connete =self.builGrap.connected_low_nodes(G)
        print("nombre des noeuds connectes dans le graphe de P_valeurs (0.05) connectee", len(list_noeuds_connete), list_noeuds_connete & valuetrue, len(list_noeuds_connete & valuetrue))

        if choice in self.choiceOfConstructGraph:
            graph_builder_func = self.choiceOfConstructGraph[choice]
            self.G = graph_builder_func(G,p_valeur,no_singleton)
            self.choice=choice
        else:
            print(f"Le choix du constructeur de graphe '{choice}' n'est pas valide.")
       
        if self.output:
            a="sauvegarde/"+self.directory+"/dataset/Vue"
            self.builGrap.save_graph(G, a+'1.txt')
            self.builGrap.save_graph(self.G, a+'2.txt')

        Y,self.model =self.compute_embedding(G,self.G) 
        self.drawCurve.draw_Single_curve(y_values=Y,
                                              title="courbe progression de l'apprentissage  ", 
                                              x_label='epoques',
                                              y_label="valeur en fonction du gratient ", 
                                             save_file="sauvegarde/"+self.directory+"/graphique_fonction/learn_function.png")

    @abstractmethod
    def compute_embedding(G: nx.Graph , G1: nx.Graph):
        """
        generer l'embbeding 

        Parameters
        ----------
        G1   :  graphe de la premier vue
        
        G2 : graphe de la deuxieme vue

        """
            

    
class ManeView(MultiView):
    def __init__(self):
      super().__init__()
      self.batch_size=500
      self.dimensions=48
      self.learning_rate=0.001
      self.epochs=10 #default 10   
      self.alpha=1.0
      self.beta= 1.0 #1.0
      self.negative_sampling=10.0   
    
    def choice_bach_size(self,input_value):
        if input_value == '1':
            return 300
        elif input_value == '2':
            return 700
        elif input_value == '3':
            return 8500
        elif input_value == '4':
            return 1300
        elif input_value == '5':
            return 10000
        elif input_value == '6':
            return 15000
        elif input_value == '7':
            return 2000
        else:
            return 0  


    def compute_embedding(self,G1,G2):
                """
                Initialisation  parametres  et entrainement du model
                """
                self.batch_size=self.choice_bach_size(self.choice)
                G=[G1,G2]
                values_lossFunction=[] 
                if torch.cuda.is_available() and not self.cuda:
                    print("WARNING: Vous disposez d'un GPU , Vous pouvez essayer  le GPU avec le parametre --cuda")
                device = 'cuda:0' if torch.cuda.is_available() and self.cuda else 'cpu'
                self.device = device
                print("Running on device: ", device)
                
                common_nodes = sorted(set(G[0]).intersection(*G))
                print('Nombre de noeud commun a toutes les vues: ', len(common_nodes))
                node2idx = {n: idx for (idx, n) in enumerate(common_nodes)}
                idx2node = {idx: n for (idx, n) in enumerate(common_nodes)}

                if self.read_pair:

                    nodes_idx_nets, neigh_idx_nets = self.read_word2vec_pairs(self.nviews)

                else:
                    nodes_idx_nets = []
                    neigh_idx_nets = []
                    for n_net in range (self.nviews): 
                        params = self.parametreNode2vec[n_net]    
                        view_id = n_net + 1
                        print("View ", view_id)
                        nodes_idx, neigh_idx = buid_views.construct_word2vec_pairs( G[n_net],
                                                                                        view_id,
                                                                                        common_nodes,
                                                                                        pvalue=params['p'],
                                                                                        qvalue=params['q'],
                                                                                        window_size=params['window_size'],
                                                                                        n_walk=params['num_walks'],
                                                                                        walk_length=params['walk_length'],  
                                                                                        output=self.output,  
                                                                                        node2idx=node2idx,
                                                                                        directory=self.directory
                                                                                    )

                        nodes_idx_nets.append(nodes_idx)
                        neigh_idx_nets.append(neigh_idx)

                multinomial_nodes_idx = self.degree_nodes_common_nodes(G, common_nodes, node2idx)

                embed_freq = Variable(torch.Tensor(multinomial_nodes_idx))

                modelMane = model_NN.ManeAI(self.nviews, self.dimensions, self.device, len(common_nodes), embed_freq, self.batch_size)
                modelMane.to(device)

                epo = 0
                min_pair_length = nodes_idx_nets[0].size
                for n_net in range(self.nviews):
                    if min_pair_length > nodes_idx_nets[n_net].size:
                        min_pair_length = nodes_idx_nets[n_net].size
                print("Le nombre total de paire de noeuds : ", min_pair_length)
                print("Debut de l'entrainement! \n")
                start_init = time.time()
                while epo <= self.epochs - 1:
                    epo += 1
                    optimizer = torch.optim.Adam(modelMane.parameters(), lr=self.learning_rate)
                    running_loss = 0
                    num_batches = 0
                    shuffle_indices_nets = []
                    fifty = False

                    for n_net in range(self.nviews):
                        shuffle_indices = [x for x in range(nodes_idx_nets[n_net].size)]
                        random.shuffle(shuffle_indices)
                        shuffle_indices_nets.append(shuffle_indices)
                    for count in range(0, min_pair_length, self.batch_size):
                        optimizer.zero_grad()
                        loss = modelMane(count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, self.alpha, self.beta)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.detach().item()
                        num_batches += 1
                        torch.cuda.empty_cache()
                        gc.collect()
                
                    total_loss = running_loss / (num_batches)
                    if epo==self.epochs:
                        elapsed = time.time() - start_init
                        print('Epoque=', epo, '\t  Total_time_est =', elapsed, ' seconds\t total_loss=', total_loss)
                    values_lossFunction.append(total_loss)
                concat_tensors = modelMane.node_embeddings[0].weight.detach().cpu()
            # print('Embedding of view ', 1, ' ', concat_tensors)

                for i_tensor in range(1, modelMane.num_net):
                    #print('Embedding of view ', (i_tensor + 1), ' ', modelMane.node_embeddings[i_tensor].weight.detach().cpu())
                    concat_tensors = torch.cat((concat_tensors, modelMane.node_embeddings[i_tensor].weight.detach().cpu()), 1)
                embedding_Multiview = np.array(concat_tensors)
                if self.output :
                    emb_file ="sauvegarde/"+self.directory+"/embedding/" + "Embedding_Entrainer_epoques_" + str(epo) + "_" + ".txt"
                    fo = open(emb_file, 'a+')
                    for idx in range(len(embedding_Multiview)):
                        word = (idx2node[idx])
                        fo.write(str(word) + ' ' + ' '.join(map(str, embedding_Multiview[idx])) + '\n')
                    fo.close()
                labels= np.array([idx2node[idx] for idx in range(len(idx2node))]).reshape(-1, 1)
                embedding_Multiview = np.hstack((labels, embedding_Multiview))

             
                return values_lossFunction,embedding_Multiview
    
    def degree_nodes_common_nodes(self, G, common_nodes, node2idx):
            """
            Assigns scores for negative sampling distribution
            """
            degrees_idx = dict((node2idx[v], 0) for v in common_nodes)
            multinomial_nodesidx = []
            for node in common_nodes:
                degrees_idx[node2idx[node]] = sum([G[n].degree(node) for n in range(len(G))])
            for node in common_nodes:
                multinomial_nodesidx.append(degrees_idx[node2idx[node]] ** (0.75))

            return multinomial_nodesidx
    
    def read_word2vec_pairs(self,nviews):
            """

            :param current_path: path for two files, one keeps only the node indices, the other keeps only the neighbor node
            indices of already generated pairs (node,neighbor), i.e, node indices and neighbor indices are kept separately.
            method "construct_word2vec_pairs" can be used to obtain these files.
            :E.g.:

            for pairs (9,2) (4,5) (8,6) one file keeps 9 4 8 the other file keeps 2 5 6.

            :param nviews: number of views
            :return: Two lists for all views, each list keeps the node indices of node pairs (node, neigh).
            nodes_idx_nets for node, neigh_idx_nets for neighbor
            """
            path="sauvegarde/fichierMane/" 
            nodes_idx_nets = []
            neigh_idx_nets = []

            for n_net in range(nviews):
                neigh_idx_nets.append(np.loadtxt(path + "Couples_ids/neighidxPairs_" + str(n_net + 1) + ".txt"))
                nodes_idx_nets.append(np.loadtxt(path + "Couples_nodes/nodesidxPairs_" + str(n_net + 1) + ".txt"))
            return nodes_idx_nets, neigh_idx_nets

   


    

class SaeView(MultiView):

    def __init__(self):
        super().__init__()
        """Declare variables."""
        self.epochs=10
        self.directed=False

        self.parametreNode2vec1=[ {'p':0.25,  'q':0.5, 'window_size':10, 'num_walks': 10, 'walk_length': 20,'dimensions':48,},
                                 {'p':1.0,  'q':1.0, 'window_size':10, 'num_walks': 10, 'walk_length': 15,'dimensions':16,},
                                ] 

    

    
    def compute_embedding(self,G1,G2):
        """
         production de l'espace vectoriels 
        """
        G=[G1,G2]
        views=self.generate_node2vec_embeddings(G,self.parametreNode2vec1)
        X_unified= self.generate_unified_X(views)
        views_probabilities1=self.calculate_P_v_ij(views,5)
        modelSae=model_NN.Sae_AI()
        Y,Y_init=modelSae.calculateY_init(X_unified,self.epochs)
        self.drawCurve.draw_Single_curve(y_values=Y,
                                              title="courbe progression de l'apprentissage du SAE ", 
                                              x_label='epoques',
                                              y_label="valeur en fonction du gratient ", 
                                             save_file="sauvegarde/"+self.directory+"/graphique_fonction/learnSAE.png")
        q_ij=modelSae.calcule_qij(Y_init)
        P_ij=modelSae.calculate_Pij(views_probabilities1,views)
        ylost,embedding=modelSae.optimisation_divergence_kl(Y_init,P_ij,q_ij,taux_apprentissage=0.01, nombre_iterations=10)
        
        #if self.output :
        #        emb_file ="sauvegarde/"+self.directory+"/embedding/" + "Embedding_Entrainer_" + str(self.epochs) + "_" + ".txt"
        #        fo = open(emb_file, 'a+')
        #        for idx in range(len(embedding)):
        #            word = (embedding[idx])
        #            fo.write(str(word) + ' ' + ' '.join(map(str,embedding[idx])) + '\n')
        #        fo.close()
        fichier="sauvegarde/"+self.directory+"/embedding/embedding.txt"
        #embedding = np.array(embedding, dtype=float)
        np.savetxt(fichier, embedding, delimiter=' ', fmt='%f')
        return ylost, embedding
    
    def generate_node2vec_embeddings(self,graphs, node2vec_params_list):
            """
            Génère des embeddings Node2Vec pour différentes vues.

            :param graphs: Liste des graphes au format networkx.Graph.
            :param node2vec_params_list: Liste des paramètres Node2Vec pour chaque graphe.
            :return: Liste des embeddings pour chaque vue.
            """
            views = []
            for G, node2vec_params in zip(graphs, node2vec_params_list):
                # Vérifiez si le graphe est du bon format
                if not isinstance(G, nx.Graph):
                    raise ValueError("Les éléments de la liste 'graphs' doivent être des objets 'networkx.Graph'.")
                
                """# Créez un modèle Node2Vec
                node2vec = Node2Vec(G, dimensions=node2vec_params['dimensions'],
                                    walk_length=node2vec_params['walk_length'],
                                    num_walks=node2vec_params['num_walks'],)
                                    #workers=node2vec_params['workers'])

                # Entraînez le modèle Node2Vec
                model = node2vec.fit(window=node2vec_params['window_size'],
                                   # min_count=node2vec_params['min_count'],
                                    #batch_words=node2vec_params['batch_words'],
                                    sg=1)
                """

                my_graph = buid_views.Graph(G, self.directed, node2vec_params['p'],node2vec_params['q'])
                my_graph.preprocess_transition_probs()
                my_walks = my_graph.simulate_walks(node2vec_params['num_walks'], node2vec_params['walk_length'])
                my_model= Word2Vec( my_walks,
                                       vector_size=node2vec_params['dimensions'],  # size=self.dimensions,
                                       window=node2vec_params['window_size'],
                                       min_count=5,
                                       negative=5,
                                       sg=1,
                                       #workers=self.workers,
                                       epochs=self.epochs,
                                      ) 




                # Générez les embeddings de chaque vue  et stockez-les
                words = my_model.wv.index_to_key
                vectors = my_model.wv.vectors
                embedding = np.column_stack((words, vectors))
                print("la forme est ", embedding.shape)
                print(embedding[:,0])
               # embeddings = [[int(node)] + list(my_embeddings.wv[str(node)]) for node in G.nodes()]
                #my_array = np.array(embeddings)
                views.append(embedding)

            return views   


    def generate_unified_X(self, views):
        """
        Génère un ensemble de données unifié en concaténant différentes vues.

        :param views: Liste des différentes vues à concaténer (tableaux NumPy).
        :return: Ensemble de données unifié (tableau NumPy).
        """
        if not all(isinstance(view, (list, np.ndarray))  for view in views):
            raise ValueError("Toutes les vues doivent être des tableaux NumPy.")

        # Obtenir toutes les étiquettes uniques
        all_labels = set().union(*[set(view[:, 0]) for view in views])
        total_features = sum(view.shape[1] - 1 for view in views) + 1
        X_unified = np.zeros((len(all_labels), total_features))

        # Initialiser la première colonne de X_unified avec les étiquettes
        X_unified[:, 0] = list(all_labels)

        start_idx = 1
        for view in views:
            # Assembler les données dans X_unified
            for row in view:
                label = row[0]
                features = row[1:]

                # Trouver l'indice de l'étiquette dans X_unified
                label_index = np.where(X_unified[:, 0] == label)[0][0]

                # Copier les caractéristiques dans X_unified
                X_unified[label_index, start_idx:start_idx + len(features)] = features

            # Mettre à jour l'indice de début pour la prochaine vue
            start_idx += view.shape[1] - 1

        return X_unified


    def calculate_P_v_ij(self,views, k):
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
              sigmas_squared = self.calculate_sigmas_squared(squared_distances, k)  # notre fonction renvoie sigma^2

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

   
    def calculate_sigmas_squared(self,squared_distances, k):
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
                1e-10,  # borne inférieure de sigma^2
                1e10  # borne supérieure de sigma^2
            )

        return sigmas_squared
###



class DbscanView(MultiView):
    def __init__(self):
      super().__init__()

    def compute_embedding(self,G1,G2,choix):
        """
         production de l'espace vectoriels 
        """
        if choix=="mane":
            dbsEmbbeding=ManeView()
            self.model=dbsEmbbeding.compute_embedding(G1,G2)
        else:
            dbsEmbbeding=SaeView()
            self.model=dbsEmbbeding.compute_embedding(G1,G2)
    
    def dbscan_cluster(self, eps=0.3, min_samples=5):
    
        # Séparation des noms des échantillons et des caractéristiques
        label_nodes= self.model[:, 0].astype(int)
        X = self.model[:, 1:].astype(float)
        
        # Normalisation des vecteurs 
        X = StandardScaler().fit_transform(X)
        
        # Configuration et exécution de DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
        
        label_cluter = db.labels_
        
        # Création d'une liste pour regrouper les échantillons par cluster
        clusters_list = []
        unique_labels = set(label_cluter) # on regroupe les labels de cluter en un representant unique 
        for label in unique_labels:
            if label != -1:  # On exclut les points de bruit
                cluster = [name for name, lbl in zip(label_nodes, label_cluter) if lbl == label]
                clusters_list.append(cluster)
        
        # Affichage de chaque cluster et des échantillons qu'il contient
        for idx, cluster in enumerate(clusters_list):
            print(f"Cluster {idx} : {cluster}")
        
        # Évaluation du modèle 
        n_clusters_ = len(clusters_list)
        n_noise_ = list(label_cluter).count(-1)
        print('nombre de clusters: %d' % n_clusters_)
        print('nombre de noeuds considerer comme bruit : %d' % n_noise_)
        if n_clusters_>1 :
            score= self.calculate_silhouette_score(X, label_cluter)
        
        return clusters_list  # Retourne la liste des clusters
    
    def evaluate_cluter(self,clusters_list, mon_ensemble):
        # Liste pour stocker les résultats
        results_list = []
        le_truehit = mon_ensemble[1]
        # Parcourir chaque cluster et trouver les éléments communs avec l'ensemble
        for cluster in clusters_list:
            common_elements = set(cluster) & le_truehit  # Intersection entre le cluster et l'ensemble
            number_of_common_elements = len(common_elements)
            
            # Ajouter le cluster, le nombre d'éléments communs, et les éléments communs à la liste des résultats
            results_list.append([cluster, number_of_common_elements, common_elements])
        
        # Trier la liste des résultats en fonction du nombre d'éléments communs, en ordre décroissant
        results_list = sorted(results_list, key=lambda x: x[1], reverse=True)
        
        # Affichage optionnel des résultats pour chaque cluster
        for result in results_list:
            print(f"le Cluster: {result[0]}, à : {result[1]}, element(s)en commun avec le module à detecter : {result[2]}")
        
        return results_list
    
    def get_most_similar(self, elt: str, number: int):
            """
            Collect similar nodes.

            Parameters
            ----------
            elt    : str
                    node identifier
            number : int
                    number of most similar nodes to collect

            """
            return 
    
    def calculate_silhouette_score(self,X, label_cluter):
        """
        Calcule le coefficient de silhouette en excluant les points de bruit.
        
        :param X: np.array, La matrice de données.
        :param label_cluter: list ou np.array, Les étiquettes de cluster attribuées par l'algorithme de clustering.
        :return: float, Le coefficient de silhouette, ou None s'il n'y a pas assez de clusters pour le calculer.
        """
        
        # Exclusion des points de bruit
        non_noise_indices = np.where(label_cluter != -1)[0]
        non_noise_labels = label_cluter[non_noise_indices]
        non_noise_X = X[non_noise_indices]
        # Calcul du coefficient de silhouette sans les points de bruit
        score = silhouette_score(non_noise_X, non_noise_labels,metric='cosine')
        return score