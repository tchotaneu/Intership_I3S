import networkx as nx
import numpy as np
from scipy.stats import norm  # Importation de norm pour résoudre le problème de référence non définie
import time
import random
import os
import matplotlib.pyplot as plt
import zipfile
import shutil
import csv
from typing import Dict

class DrawCurve():

    """
 
    Classe pour dessiner les courbes ou les illustration .
    
    """
    def draw_Single_curve(self, y_values=None, title="", x_label="", y_label="", save_file=None):
        if  not y_values:
            print("y_values est  impossible de tracer la courbe.")
            return
        x_values = list(range(len(y_values)))
        """
        Dessine une courbe et la sauvegarde éventuellement dans un fichier.
        
        :x_values: Liste ou objet similaire contenant les valeurs de l'axe des x.
        :param y_values: Liste contenant les valeurs de l'axe des y.
        :param title: Titre du graphique.
        :param x_label: Étiquette pour l'axe des x.
        :param y_label: Étiquette pour l'axe des y.
        :param save_file: Optionnel; si fourni, le chemin où le graphique doit être sauvegardé.
        """
        x_values = list(range(len(y_values)))
        plt.figure(figsize=(10,6))  
        plt.plot(x_values, y_values, marker='o')  # Ajout d'un marqueur pour une meilleure visualisation
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)  # Ajout d'une grille pour une meilleure lisibilité du graphique
        
        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))  # Crée les répertoires dans le chemin s'ils n'existent pas
            plt.savefig(save_file, dpi=300)
            print(f"Graphique sauvegardé à {save_file}")
        else:
            plt.show()  # Affiche le graphique si save_file n'est pas fourni
        
        plt.close()


    def draw_Multi_curves(self, x_values, y_values_list, title, x_label, y_label, legends=None, save_file=None):
        """
        Dessine plusieurs courbes sur le même graphique et le sauvegarde éventuellement dans un fichier.
        
        :param x_values: Liste ou objet similaire contenant les valeurs de l'axe des x communes à toutes les courbes.
        :param y_values_list: Liste de listes ou objets similaires contenant les valeurs de l'axe des y pour chaque courbe.
        :param title: Titre du graphique.
        :param x_label: Étiquette pour l'axe des x.
        :param y_label: Étiquette pour l'axe des y.
        :param legends: Liste de légendes pour chaque courbe.
        :param save_file: Optionnel; si fourni, le chemin où le graphique doit être sauvegardé.
        """
        if not x_values or not y_values_list:
            print("x_values ou y_values_list sont vides. Impossible de tracer les courbes.")
            return
        
        plt.figure(figsize=(10,6))  # Vous pouvez ajuster la taille de la figure selon vos besoins
        
        for i, y_values in enumerate(y_values_list):
            if legends and len(legends) > i:
                plt.plot(x_values, y_values, marker='o', label=legends[i])  # Avec légende
            else:
                plt.plot(x_values, y_values, marker='o')  # Sans légende
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)  # Ajout d'une grille pour une meilleure lisibilité du graphique
        
        if legends:
            plt.legend(loc='best')  # Affiche la légende à la meilleure position
        
        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))  # Crée les répertoires dans le chemin s'ils n'existent pas
            plt.savefig(save_file, dpi=300)
            print(f"Graphique sauvegardé à {save_file}")
        else:
            plt.show()  # Affiche le graphique si save_file n'est pas fourni
        
        plt.close()



class GraphBuilder():
    """
    Classe pour construire différents types de graphes basés sur un graphe d'origine.
    """
    def construct_graph9(self, G: nx.Graph, p_value: float = 0.05) -> nx.Graph:
        """
        Construit un nouveau graphe basé sur le graphe d'origine G, en extrayant les composantes connexes
        des nœuds avec un poids inférieur à p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
    
        # Créer un nouveau graphe
        G_prime = nx.Graph()

        # Parcourir toutes les arêtes dans G et les ajouter à G_prime si les deux nœuds ont une p_value <= p_value
        for u, v, data in G.edges(data=True):
            u_data = G.nodes[u]["weight"]
            v_data = G.nodes[v]["weight"]
            
            if u_data <= p_value and v_data <= p_value:
                G_prime.add_edge(u, v, **data)
                G_prime.nodes[u]["weight"] = u_data
                G_prime.nodes[v]["weight"] = v_data

            return G_prime

    def construct_graph1(self, G: nx.Graph, p_value: float = 0.05,no_singletons:bool = True) -> nx.Graph:
        """
        Construit un nouveau graphe basé sur le graphe d'origine G, en extrayant les composantes connexes
        des nœuds avec un poids inférieur à p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
   
            # Créer un nouveau graphe
        G_prime = nx.Graph()
            
            # Ajouter tous les nœuds de G à G_prime
        G_prime.add_nodes_from(G.nodes(data=True))  
            # Parcourir toutes les arêtes dans G et les ajouter à G_prime si les deux nœuds ont une p_value < 0.05
        for u, v,data in G.edges(data=True):
            u_data = G.nodes[u]["weight"]
            v_data = G.nodes[v]["weight"]
                
            if u_data <= p_value and v_data <= p_value:
                G_prime.add_edge(u, v, **data)
        if no_singletons:
        # Exclure les nœuds singletons
            singletons = [node for node in G_prime.nodes if G_prime.degree(node) == 0]
            G_prime.remove_nodes_from(singletons)
        
        return G_prime
        
    
    def construct_graph2(self, G: nx.Graph, p_value: float = 0.05 ,no_singleton:bool=True) -> nx.Graph:
        """
        Construit un graphe complet entre les nœuds avec un poids inférieur à p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        # Ajouter tous les nœuds de G à G_prime
        #nodes_singletons = [(node, data) for node, data in G.nodes(data=True) if data['weight'] > p_value]
        nodes_pvalue = [node for node, data in G.nodes(data=True) if data['weight'] <= p_value]

        G_prime.add_edges_from((u, v) for i, u in enumerate(nodes_pvalue) for v in nodes_pvalue[i + 1:])
        if no_singleton:
        # Exclure les nœuds singletons
            singletons = [node for node in G_prime.nodes if G_prime.degree(node) == 0]
            G_prime.remove_nodes_from(singletons)
        return G_prime
    
    def construct_graph3(self, G: nx.Graph, p_value: float = 0.05,no_singletons:bool = True) -> nx.Graph:
        """
        Construit un graphe en ajoutant des arêtes entre les nœuds avec un poids inférieur à p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        G_prime.add_edges_from(G.edges(data=True))
        
        p_nodes = [node for node, data in G.nodes(data=True) if data['weight'] < p_value]
        G_prime.add_edges_from((u, v) for i, u in enumerate(p_nodes) for v in p_nodes[i + 1:] if not G_prime.has_edge(u, v))

        if no_singletons:
        # Exclure les nœuds singletons
            singletons = [node for node in G_prime.nodes if G_prime.degree(node) == 0]
            G_prime.remove_nodes_from(singletons)
        
        return G_prime
    
    def construct_graph4(self, G: nx.Graph, quantile: float = 0.4, no_singletons:bool = True) -> nx.Graph:
        """
        Construit un graphe en ajoutant des arêtes entre les nœuds dont la différence de "Z_Scores" est inférieure à 0.4.
        
        :param G: Le graphe d'origine.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        
        quantiles = {node: min(10, norm.ppf(1 - data['weight'])) for node, data in G.nodes(data=True)}
        G_prime.add_edges_from((u, v) for u, v in G.edges() if abs(quantiles[u] - quantiles[v]) < quantile)
        if no_singletons:
        # Exclure les nœuds singletons
            singletons = [node for node in G_prime.nodes if G_prime.degree(node) == 0]
            G_prime.remove_nodes_from(singletons)
        
        return G_prime
    
    def construct_graph5(self, G: nx.Graph, p_value: float = 0.05) -> nx.Graph:
        """
        Construit un graphe en créant une composante connexe pour les nœuds avec un poids supérieur à p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        
        high_p_nodes = [node for node, data in G.nodes(data=True) if data['weight'] > p_value]
        G_prime.add_edges_from((u, v) for i, u in enumerate(high_p_nodes) for v in high_p_nodes[i + 1:])
        
        return G_prime

    
    def construct_graph6(self, G: nx.Graph, p_value: float = 0.05) -> nx.Graph:
        """
        Construit un graphe en créant des arêtes entre les nœuds avec poids >= p_value.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        
        G_prime.add_edges_from((u, v) for u, v in G.edges() if G.nodes[u]['weight'] >= p_value and G.nodes[v]['weight'] >= p_value)
        
        return G_prime
    

    
    def construct_graph7(self, G: nx.Graph, p_value: float = 0.05,no_singletons:bool = True) -> nx.Graph:
        """
        Construit un graphe en créant des composantes connexes complètes pour les nœuds avec des poids faibles et forts.
        
        :param G: Le graphe d'origine.
        :param p_value: La valeur de p utilisée pour filtrer les nœuds.
        :return: Le nouveau graphe construit.
        """
        G_prime = nx.Graph()
        G_prime.add_nodes_from(G.nodes(data=True))
        
        low_p_nodes = [node for node, data in G.nodes(data=True) if data['weight'] <= p_value]
        high_p_nodes = [node for node, data in G.nodes(data=True) if data['weight'] > p_value]
        
        G_prime.add_edges_from((u, v) for i, u in enumerate(low_p_nodes) for v in low_p_nodes[i + 1:])
        G_prime.add_edges_from((u, v) for i, u in enumerate(high_p_nodes) for v in high_p_nodes[i + 1:])

        
        return G_prime

    def isolated_low_nodes(self,G: nx.Graph, pvalue=0.05):
        isolated_low_pvalue_nodes = []
        
        for node, data in G.nodes(data=True):
            # Vérifie si la p-value est inférieure au seuil
            if "weight" in data and data["weight"] < pvalue:
                is_isolated = True
                # Vérifie si le nœud est connecté à d'autres nœuds avec une p-value inférieure
                for neighbor in G.neighbors(node):
                    if "weight" in G.nodes[neighbor] and G.nodes[neighbor]["weight"] < pvalue:
                        is_isolated = False
                        break
                if is_isolated:
                    isolated_low_pvalue_nodes.append(node)
        
        return set(isolated_low_pvalue_nodes)
    
    def low_nodes(self, G, pvalue=0.05):
        low_pvalue_nodes = []

        for node, data in G.nodes(data=True):
            # Vérifie si la p-value est inférieure au seuil
            if "weight" in data and data["weight"] <= pvalue:
                low_pvalue_nodes.append(node)

        return set(low_pvalue_nodes)
    
    def connected_low_nodes(self,G, threshold=0.05):
        connected_low_pvalue_nodes = []

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            low_pvalue_neighbors = [
                neighbor for neighbor in neighbors
                if "weight" in G.nodes[neighbor] and G.nodes[neighbor]["weight"] <= threshold
            ]

            if len(low_pvalue_neighbors) > 0:
                connected_low_pvalue_nodes.append(node)

        return set(connected_low_pvalue_nodes)
    

    def get_low_pvalue_nodes_with_low_pvalue_neighbors(self,G, threshold=0.05):
        low_pvalue_nodes = [node for node in G.nodes() if "weight" in G.nodes[node] and G.nodes[node]["weight"] < threshold]
        result_nodes = []

        for node in low_pvalue_nodes:
            neighbors = list(G.neighbors(node))
            low_pvalue_neighbors = [
                neighbor for neighbor in neighbors
                if "weight" in G.nodes[neighbor] and G.nodes[neighbor]["weight"] <= threshold
            ]

            if low_pvalue_neighbors:
                result_nodes.append(node)

        return result_nodes


    
    def load_graph(self,filename: str) -> nx.Graph:
        G_reconstructed = nx.Graph()
        with open(filename, 'r') as f:
            mode = None  # Utilisé pour savoir si nous lisons un nœud ou une arête
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    mode = line[2:].lower()  # Définir le mode sur 'nodes' ou 'edges' en fonction du commentaire
                    continue
                
                parts = line.split()
                if mode == 'nodes':
                    _, node, weight = parts
                    G_reconstructed.add_node(node, weight=int(weight))
                elif mode == 'edges':
                    _, u, v, weight = parts
                    G_reconstructed.add_edge(u, v, weight=int(weight))
        return G_reconstructed
    
    def save_graph(self,G: nx.Graph, filename: str):
        with open(filename, 'w') as f:
            # Écrire les nœuds et leurs attributs
            f.write("# Nodes\n")
            for node, data in G.nodes(data=True):
                weight = data.get('weight', 1)  # Utilisez 1 comme poids par défaut si aucun poids n'est spécifié
                f.write(f"node {node} {weight}\n")
            
            # Écrire les arêtes
            f.write("# Edges\n")
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)  # Utilisez 1 comme poids par défaut si aucun poids n'est spécifié pour les arêtes
                f.write(f"edge {u} {v} {weight}\n")



class Archiver():

    def __init__(self):
        """Declare variables."""
        self.source= "sauvegarde/"
        self.dest = "sauvegarde/fichierZip/"

    def add_to_zip(self, zip_filename: str, folder_path: str):
        """
        Ajoute un dossier à une archive zip.
        
        :param zip_filename: Le chemin de l'archive zip.
        :param folder_path: Le chemin du dossier à ajouter à l'archive zip.
        """
        try:
            # Vérifie si le dossier à ajouter existe
            if not os.path.exists(folder_path):
                print(f"Le dossier {folder_path} n'existe pas.")
                return
            
            # Ouvre ou crée l'archive zip
            with zipfile.ZipFile(zip_filename, 'a') as zipf:
                # Obtient le nom du dossier parent
                parent_folder_name = os.path.basename(folder_path)
                
                # Parcourt le dossier et ses sous-dossiers et fichiers
                for foldername, subfolders, filenames in os.walk(folder_path):
                    for filename in filenames:
                        # Construit le chemin complet du fichier
                        file_path = os.path.join(foldername, filename)
                        
                        # Construit le nom d'arc avec le nom du dossier parent
                        arcname = os.path.join(parent_folder_name, os.path.relpath(file_path, folder_path))
                        
                        # Ajoute le fichier à l'archive zip
                        zipf.write(file_path, arcname=arcname)
            
            print(f"Le dossier {folder_path} a été ajouté avec succès à {zip_filename}.")
            
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'ajout du dossier à l'archive zip : {e}")



    def copy_directory(self,src: str, dest_dir: str):
        """
        Copie le contenu du dossier source vers le dossier de destination.
        
        :param src_dir: Le chemin du dossier source.
        :param dest_dir: Le chemin du dossier de destination.
        """
        src_dir=src
       
        try:
            # Vérifie si le dossier source existe
            if not os.path.exists(src_dir):
                print(f"Le dossier source {src_dir} n'existe pas.")
                return
            
            # Crée le dossier de destination s'il n'existe pas
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Copie le contenu du dossier source vers le dossier de destination
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest_dir, item)
                
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)
            print(f"Le contenu de {src_dir} a été copié avec succès vers {dest_dir}.")
            
        except Exception as e:
            print(f"Une erreur s'est produite lors de la copie des dossiers : {e}")


    def save_in_csv(self, chemin_fichier: str, parametres: Dict[str, str]) -> None:
        """
        Insère les valeurs des paramètres dans un fichier CSV existant.
        
        :param chemin_fichier: Le chemin du fichier CSV.
        :param parametres: Un dictionnaire contenant les paramètres à insérer.
        """
        # Vérifier si le fichier CSV existe déjà
        try:
            with open(chemin_fichier, 'r') as fichier:
                lecteur = csv.DictReader(fichier)
                champs = lecteur.fieldnames  # Récupérer les noms des colonnes existantes
        except FileNotFoundError:
            champs = list(parametres.keys())  # Utiliser les clés du dictionnaire comme noms de colonnes si le fichier n'existe pas
        
        # Ouvrir le fichier en mode append ('a') pour ajouter une nouvelle ligne
        with open(chemin_fichier, 'a', newline='') as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=champs)
            
            # Écrire les noms des colonnes si le fichier est vide
            if fichier.tell() == 0:
                ecrivain.writeheader()
            
            # Insérer les paramètres dans une nouvelle ligne du fichier CSV
            ecrivain.writerow(parametres)

    def deleteFileInDirectory(self, my_directory):
        """
        Supprime tous les fichiers dans le dossier spécifié.
        
        :param my_directory: Le chemin du dossier dont vous voulez supprimer tous les fichiers.
        """
        # Vérifie si le chemin donné est un dossier valide
        if not os.path.isdir(my_directory):
            print(f"{my_directory} n'est pas un dossier valide.")
            return
        
        # Liste tous les fichiers dans le dossier
        for filename in os.listdir(my_directory):
            file_path = os.path.join(my_directory, filename)
            
            # Vérifie si c'est un fichier et non un dossier, puis le supprime
            if os.path.isfile(file_path):
                os.remove(file_path)
                #print(f"Fichier {filename} supprimé avec succès.")
            else:
                print(f"{filename} n'est pas un fichier et ne sera pas supprimé.")



    def delete_directory(self, repertoire, verbose=True):
        """
        Supprime le répertoire spécifié de manière récursive.

        Cette fonction tente de supprimer le répertoire spécifié, y compris tout son contenu,
        de manière récursive. En cas de réussite, un message indiquant la suppression
        réussie est affiché, sauf si le mode silencieux est activé. En cas d'erreur lors de la suppression, un message d'erreur
        est affiché, fournissant des informations sur la nature de l'erreur.
        Args:
            repertoire (str): Le chemin vers le répertoire que vous souhaitez supprimer.
            silencieux (bool, optional): Si True, aucune sortie ne sera affichée (par défaut False).

        """
        if not os.path.exists(repertoire):
            raise FileNotFoundError(f"Le répertoire {repertoire} n'existe pas.")
        
        try:
            shutil.rmtree(repertoire)
            if not verbose:
                print(f"Le répertoire {repertoire} a été supprimé avec succès.")
        except OSError as e:
            print(f"Erreur lors de la suppression du répertoire {repertoire}: {e}")


    def delete_file_ifexist(self,nom_fichier, nom_repertoire):
        chemin_fichier = os.path.join(nom_repertoire, nom_fichier)

        if os.path.isfile(chemin_fichier):
            os.remove(chemin_fichier)
            print(f"Le fichier {nom_fichier} a été supprimé.")
        else:
            print(f"Le fichier {nom_fichier} n'existe pas dans le répertoire {nom_repertoire}.")
    

    

    def create_model_directories(self,base_directory, model_name,module_size, metric=None,graphe=None,sing=None):
        rep_save = os.path.join(base_directory,model_name,model_name)
        
        if metric:
            rep_save += f"_{metric[:3]}"
        
        rep_save += f"_size_{module_size}"
        if graphe:
             rep_save += f"_gra_{graphe}"
        if sing:
             rep_save += f"_NoSin_{sing}"
        
        directories_to_create = [
            "dataset",
            "embedding",
            "resultats"
        ]
        if model_name == 'SaeView':
            directories_to_create.extend([
                "graphique_fonction"
            ])
        elif model_name == 'ManeView':
            directories_to_create.extend([
                "graphique_fonction",
                "Couples_ids",
                "Couples_nodes",
                "Marches",
                "Paires"
            ])
        
        for directory_name in directories_to_create:
            directory_path = os.path.join(rep_save, directory_name)
            os.makedirs(directory_path, exist_ok=True)
           # print(f"Le répertoire '{directory_path}' a été créé.")

        return rep_save



    # Exemple d'utilisation :
    #noms = ["/chemin/vers/repertoire1", "/chemin/vers/repertoire2", "/chemin/vers/repertoire3"]
    #creer_repertoires(*noms)


    # Exemple d'utilisation :
    #nom_fichier = "mon_fichier.txt"
    #nom_repertoire = "/chemin/vers/le/repertoire"
    #supprimer_fichier_dans_repertoire(nom_fichier, nom_repertoire)




# reference:
# https://github.com/sezinata/MANE.git


def save_walks(walks, out_file, elapsed):
    """
    Sauvegarde les marches aleatoire de walks node2vec  de Mane.
    """

    with open(out_file, "w") as f_out:
        for walk in walks:
            f_out.write(" ".join(map(str, walk)) + "\n")
        print("durée de la marche: ", elapsed, " seconds.\n")

    return


def save_pairs(pairs, out_file, elapsed):
    """
    Sauvegarde les  pairs de  word2vec de Mane.
    """
    with open(out_file, "w") as f_out:
        for pair in pairs:
            f_out.write(" ".join(map(str, pair)) + "\n")
    return


def save_train_neigh(pair_node, out_file):
    with open(out_file, 'w') as f:
        f.write(" ".join(map(str, pair_node)))

    return


def construct_word2vec_pairs(G, view_id, common_nodes, pvalue, qvalue, window_size, n_walk, walk_length, output,
                             node2idx,directory):
    """
    Generer et sauvegarder les pairs de noeud  Word2Vec 
    """
    if output:
        path = directory #######################"""###################"
    list_neigh = []
    G_ = Graph(G, False, pvalue, qvalue)
    G_.preprocess_transition_probs()
    start_time = time.time()
    walks = G_.simulate_walks(n_walk,
                              walk_length)
    end = time.time()
    walk_file = path + "/Marches/Walks_" + str(view_id) + ".txt"
    elapsed = end - start_time
    save_walks(walks, walk_file, elapsed)
    
    
    start_time = time.time()
    for walk in walks:
        for pos, word in enumerate(walk):
            reduced_window = random.randint(1, window_size)
            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - window_size + reduced_window)
            for pos2, word2 in enumerate(walk[start:(pos + window_size + 1 - reduced_window)], start):
                #print("listesshort" , (walk[start:(pos + window_size + 1 - reduced_window)], start))
                # don't train on the `word` itself
                if word != word2:
                    list_neigh.append((node2idx[word], node2idx[word2]))
    pair_file = path + "/Paires/Pairs_" + str(view_id) + ".txt"
    list_neigh.sort(key=lambda x: x[0])  # sorted based on keys
    list_neigh = np.array(list_neigh)
    save_pairs(list_neigh, pair_file, elapsed)

    nodes_idx, neigh_idx = zip(*[(tupl[0], tupl[1]) for tupl in list_neigh])  # gives tuple
    nodesidx_file = path + "/Couples_ids/nodesidxPairs_" + str(view_id) + ".txt"
    save_train_neigh(np.array(list(nodes_idx)), nodesidx_file)

    neigh_idx_file = path + "/Couples_nodes/neighidxPairs_" + str(view_id) + ".txt"
    save_train_neigh(np.array(list(neigh_idx)), neigh_idx_file)
    end = time.time()

    elapsed = end - start_time
    print("Temps écoulé durant la contruction des paires pour le réseau " + str(view_id) + ": ", elapsed, " seconds.\n")

    return np.array(list(nodes_idx)), np.array(list(neigh_idx))


# reference:
# https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		#print('Walk iteration:')
		for walk_iter in range(num_walks):
			#print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks
#Sampling a new node in the walk can be efficiently done in O(1) time using alias sampling
	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(1/p) # 1 modified
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(1) # 1 modified
			else:
				unnormalized_probs.append(1/q)  #1 modified
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [1 for nbr in sorted(G.neighbors(node))] #1 modified
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
	    return J[kk]