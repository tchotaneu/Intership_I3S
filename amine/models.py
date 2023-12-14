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

import pathlib
from abc import ABC, abstractmethod
from typing import Union, Iterable
from scipy.stats import pearsonr
import networkx as nx
import numpy as np
from gensim.models import FastText, Word2Vec
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from .dimension_reduction import node2vec
from .dimension_reduction.pecanpy import node2vec as n2v
from .buid_views import GraphBuilder,Pair_nodes,DrawCurve,Save_ouput
from .model_NN import   ManeAI,Sae_AI
import torch
import time
import random
import gc
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
        self.savedirectory="amine/output"
        self.singleton=True
        self.vue1 = nx.Graph()
        self.vue2 = nx.Graph()
        self.pair=Pair_nodes()
        self.save=Save_ouput()
        self.constrution=GraphBuilder()
        self.dessin_courbe=DrawCurve()
        self.bias_attenuation=0.75
        self.dimensions=48
        self.epochs=12
        self.negative_sampling=15
        self.learning_rate=0.001
        self.alpha=1
        self.beta=1
        self.metrique="pearson" #  "pearson" =1 , "cosinus"=2 ,  "eucludian"=3
        self.directory = None
        self.model = None
        self.device = None
        self.cuda=True
        self.read_pair=False # True si nous voulons utiliser les fichiers de la derniers operations 
        self.output= True # true si nous voulons sauvegarde les fichiers 
        self.nviews=2
        self.parametreNode2vec=[ {'p':1,  'q':1, 'window_size':10, 'num_walks': 50, 'walk_length': 200, },
                                 {'p':1,  'q':1, 'window_size':10, 'num_walks': 30, 'walk_length': 150, }, ]  
        
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
            indices_similaires = self.use_similar_cosine(vectors , reference_vector)
        elif self.metrique=="pearson":
            indices_similaires=self.use_similar_pearson(vectors , reference_vector)
        elif self.metrique=="eucludian":
            indices_similaires=self.use_similar_euclidean(vectors , reference_vector)
        else :
            print("metrique non defini ")
         # Exclure le nœud de référence lui-même de la liste des nœuds similaires
        indices_similaires = [i for i in indices_similaires if i != indice_label]
        labels_similaires = labels_vectors [indices_similaires]
        
        return list(map(int, labels_similaires.tolist()))
    

    def use_similar_cosine(self, vectors, reference_vector):
        """
        Trouve les vecteurs les plus similaires au vecteur de référence en utilisant la similarité cosinus.

        :param vectors: Tableau de vecteurs à comparer.
        :type vectors: np.ndarray
        :param reference_vector: Vecteur de référence pour la comparaison.
        :type reference_vector: np.ndarray
        :return: Les indices des vecteurs dans l'ordre décroissant de similarité cosinus par rapport au vecteur de référence.
        :rtype: np.ndarray
        """
        # Normaliser le vecteur de référence
        reference_vector = reference_vector / np.linalg.norm(reference_vector)
        # Normaliser tous les vecteurs dans le tableau
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        # Calculer la similarité cosinus entre le vecteur de référence et tous les vecteurs
        similarities = cosine_similarity([reference_vector], vectors).flatten()
        # Trier les indices des vecteurs en fonction de leur similarité cosinus (en ordre décroissant)
        similar_indices = np.argsort(similarities)[::-1]

        return similar_indices


    def use_similar_pearson(self, vectors, reference_vector):
        """
        Trouve les vecteurs les plus similaires au vecteur de référence en utilisant la corrélation de Pearson.

        :param vectors: Tableau de vecteurs à comparer.
        :type vectors: np.ndarray
        :param reference_vector: Vecteur de référence pour la comparaison.
        :type reference_vector: np.ndarray
        :return: Les indices des vecteurs dans l'ordre décroissant de corrélation de Pearson par rapport au vecteur de référence.
        :rtype: np.ndarray
        """
        # Calculer la corrélation de Pearson entre le vecteur de référence et tous les vecteurs
        correlations = np.array([pearsonr(reference_vector, vector)[0] for vector in vectors])
        # Trier les indices des vecteurs en fonction de leur corrélation (en ordre décroissant)
        similar_indices = np.argsort(correlations)[::-1]
        return similar_indices


    def use_similar_euclidean(self, vectors, reference_vector):
        """
            Trouve les vecteurs les plus similaires au vecteur de référence en utilisant la distance euclidienne.

            :param vectors: Tableau de vecteurs à comparer.
            :type vectors: np.ndarray
            :param reference_vector: Vecteur de référence pour la comparaison.
            :type reference_vector: np.ndarray
            :return: Les indices des vecteurs dans l'ordre croissant de la distance euclidienne par rapport au vecteur de référence.
            :rtype: np.ndarray
        """
        # Calculer la distance euclidienne entre le vecteur de référence et tous les vecteurs
        distances = np.linalg.norm(vectors - reference_vector, axis=1)

        # Trier les indices des vecteurs en fonction de leur distance euclidienne (en ordre croissant)
        similar_indices = np.argsort(distances)

        return similar_indices

    
    def init(self, G: nx.Graph):
        self.vue1=G  
        self.vue2 =self.constrution.construct_graph1(G)
        self.save.save_graph(self.vue1,self.savedirectory+"/dataset/graphe1.txt")
        self.save.save_graph(self.vue2,self.savedirectory+"/dataset/graphe2.txt")
        Y,self.model =self.compute_embedding(self.vue1,self.vue2) 
        self.dessin_courbe.draw_Single_curve(Y,
                                             title="courbe d'apprentissage de la fonction de perte ", 
                                             x_label="epoques",
                                             y_label="loss periodes", 
                                             save_file=self.savedirectory+"/drawCurve/loss_function.png")
 
  
    def compute_embedding(self, vue1: nx.Graph ,vue2: nx.Graph):
        """
          initialisation  parametres  et entrainement du model
          generer l'embbeding 
          Parameters
          G1   :  graphe de la premier vue
          G2 : graphe de la deuxieme vue
            
        """    
        G=[vue1,vue2]
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
                nodes_idx, neigh_idx = self.pair.construct_word2vec_pairs(G=G[n_net],
                                                                            view_id=view_id,
                                                                            pvalue=params['p'],
                                                                            qvalue=params['q'],
                                                                            window_size=params['window_size'],
                                                                            n_walk=params['num_walks'],
                                                                            walk_length=params['walk_length'],  
                                                                            output=self.output,  
                                                                            node2idx=node2idx,
                                                                            directory=self.savedirectory
                                                                                        )
                        
                nodes_idx_nets.append(nodes_idx)
                neigh_idx_nets.append(neigh_idx)

        multinomial_nodes_idx = self.degree_nodes_common_nodes(G, common_nodes, node2idx)
        embed_freq = torch.Tensor(multinomial_nodes_idx)
        epo = 0
        min_pair_length = nodes_idx_nets[0].size
        for n_net in range(self.nviews):
            if min_pair_length > nodes_idx_nets[n_net].size:
                min_pair_length = nodes_idx_nets[n_net].size
        print("Le nombre total de paire de noeuds : ", min_pair_length)
        self.common_pair_nodes_views =min_pair_length
        self.batch_size=self.choice_bach_size(self.common_pair_nodes_views )
        print("la taille du bacth est : ", self.batch_size)
        modelMane = ManeAI(self.nviews, self.dimensions, self.device, len(common_nodes), embed_freq, self.batch_size, self.negative_sampling)
        modelMane.to(device)
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
            emb_file =self.savedirectory+"/embedding/" + "Embedding_Entrainer_epoques_" + str(epo) + "_" + ".txt"
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
        Attribue des scores pour la distribution d'échantillonnage négative
        """
        degrees_idx = dict((node2idx[v], 0) for v in common_nodes)
        multinomial_nodesidx = []
        for node in common_nodes:
            degrees_idx[node2idx[node]] = sum([G[n].degree(node) for n in range(len(G))])
        for node in common_nodes:
            multinomial_nodesidx.append(degrees_idx[node2idx[node]] ** (self.bias_attenuation))

        return multinomial_nodesidx          

    def choice_bach_size(self, input_value):
        if input_value <= 30000:
            return 100
        elif 30000 < input_value <= 40000:
            return 150
        elif 40000 < input_value <= 60000:
            return 250
        elif 60000 < input_value <= 100000:
            return 350
        elif 100000 < input_value <= 200000:
            return 500
        elif 200000 < input_value <= 350000:
            return 700
        elif 350000 < input_value <= 550000:
            return 1000
        elif 500000 < input_value <= 1000000:
            return 1500
        else:
            return 5000


