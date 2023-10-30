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

Entry point for the processing of real networks.
"""
import networkx as nx
import argparse
import statistics
import time
from random import Random
import sys # 

import numpy as np
#importation des fichiers Models 
import models
#import modelMane

from datasets import Datasets
from module_detection import ModuleDetection
from scores import Scores
from buid_views import Archiver
import zipfile

def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--nbruns",
        dest="number_of_runs",
        type=int,
        required=False,
        default=1000,  #1000 par defaut 
        help="specifies the number of runs (default=1000)",
    )
    parser.add_argument(
        "-n",
        "--nbmodules",
        dest="nb_modules",
        type=int,
        required=False,
        default=1,
        help="specifies the number of modules to generate (default=1)",
    )
    parser.add_argument(
        "-m",
        "--modulesize",
        dest="module_size",
        type=int,
        required=False,
        default=10,
        help="specifies the size of each modules (default=10)",
    )
    parser.add_argument(
        "-t",
        "--targetmodulesize",
        dest="target_module_size",
        type=int,
        required=False,
        default=None,
        help="specifies the expected size of each modules (default=None, i.e. no expected size)",
    )
    parser.add_argument(
        "-s",
        "--networksize",
        dest="network_size",
        type=int,
        required=False,
        default=1000,
        help="specifies the number of vertices in graph (default=1000)",
    )
    parser.add_argument(
        "-d",
        "--deledges",
        dest="removed_edges",
        type=float,
        required=False,
        default=0.0,
        help="proportion of edges to remove (default = 0.0",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        action="store_true",
        default=True,
        help="displays results on screen",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        required=False,
        default=None,
        help="name of the output file (default=no output)",
    )
    parser.add_argument(
        "-g",
        "--graphgen",
        dest="graph_generation",
        required=False,
        choices=["guyondata", "guyon", "batra", "gencat"],
        help="""Specify the data to use.
                If 'guyondata' is chosen the predictions are done on the dataset specified in
                    Guyon's paper and all other graph parameters are ignored.
                If 'guyon' is chosen, the method described in Guyon's paper is used.
                If 'batra' is chosen, data are generated using the method described in Batra's
                    paper. In this case, additional parameters might be specified:
                    param_k, param_mean_bg, param_mean_fg, param_std_bg and param_std_bg.
                    If 'gencat' is chosen, data are generated using the gencat method.""",
    )
    parser.add_argument(
        "--param_k",
        dest="param_k",
        required=False,
        default=2,
        help="""K parameter used by Batra method to specify the average shortest path length
                in a module (default=2).""",
    )
    parser.add_argument(
        "--param_mean_bg",
        dest="param_mean_bg",
        required=False,
        default=0,
        help="mean of the background used by Batra method (default=0)",
    )
    parser.add_argument(
        "--param_mean_fg",
        dest="param_mean_fg",
        required=False,
        default=1,
        help="mean of the foreground used by Batra method (default=1)",
    )
    parser.add_argument(
        "--param_std_bg",
        dest="param_std_bg",
        required=False,
        default=0,
        help="mean of the background used by Batra method (default=0)",
    )
    parser.add_argument(
        "--param_std_fg",
        dest="param_std_fg",
        required=False,
        default=1,
        help="mean of the foreground used by Batra method (default=1)",
    )
    parser.add_argument(
        '--my_model', 
        dest="my_model",
        type=str, 
        choices=['Amine', 'ManeView', 'SaeView','Dbscan'],
        required=True, 
        help='Le modèle à utiliser.'
        )

    parser.add_argument(
        '-graphe', 
        '--constructGraph', 
        dest="graphe",
        type=str, 
        choices=['1', '2', '3','4','5','6','7'],
        required=False, 
        help='La contruction de la deuxieme vue '
        )
    parser.add_argument(
        '-no_sing', 
        '--no_singleton', 
        dest="no_singleton",
        type=bool, 
        default=False,
        required=False, 
        help='La contruction de la deuxieme vue avec les noeuds singletons '
        )
    parser.add_argument(
        "-me", 
        "--metrique", 
        dest="metrique",
        type=str, 
        choices=['cosinus', 'pearson', 'eucludian'],
        required=False, 
        help='Le choix de la metrique '
        )
    parser.add_argument(
        "--pvalue",  
        dest="pvalue",
        type=float, 
        default=0.05,
        required=False, 
        help='Le choix de la p_value '

        )
    parser.add_argument(
        "--pipe",  
        dest="pipeline",
        type=bool, 
        default=False,
        required=False, 
        help="pour la reduction en de  l'espace vectoriel en deux dimension uniquemen pour le model Dbscam "
        )
    return parser.parse_args()



if __name__ == "__main__":
    # Entry point.
    arg = parse_arguments()
    if arg.my_model != 'Amine' and (arg.graphe is None or arg.metrique is None):
           print("Si le modèle n'est pas 'Amine', vous devez spécifier les arguments 'graphe' et 'metrique'.")
    """ 
    +-------------------------------------------------------------------------------------+
    We could use : 
     ---------> Node2vec() model with    --my_model Amine
     ---------> ManeView() model  with   --my_model ManeView  -me {cosinus,eucludian,pearson}  -graphe {1,2,3,4} 
     ---------> SaeView() model  with    --my_model SaeView   -me {cosinus,eucludian,pearson}  -graphe {1,2}  --no_sing True  # le calcul est trop important lorsque la deuxieme vue n'a pas de donné manquant .
     ---------> Dbscan()  model  with    --my_model Dbscan    -me{cosinus,eucludian}  --embedding {ManeView ,SaeView }  -graphe {1,2,3,4,5,6,7} --pipeline {False ,True } 

    +-------------------------------------------------------------------------------------+

    Entry point for the processing of real networks.
    """  
    # pour charger modele 

    
    # Use aggregation zscore as fitness function.
    fitness_fun = lambda the_graph, clus: Scores.aggregation_from_pvalue(
        the_graph, clus, "weight"
    )

    # set characteristic of dense network
    P_PROB = 0.09
    Q_PROB = 0.70
    NB_INITIAL_NODES = 3

    f1_scores = []
    if arg.outfile:
        outfile = open(arg.outfile, "w")
        outfile.write("#graph,time(s),nb found,real size,true hits,pvalue\n")
    for ctr in range(arg.number_of_runs):
        if arg.graph_generation == "guyondata":
            G = Datasets.get_guyon_graph(ctr + 1)
        elif arg.graph_generation == "gencat":
            G = Datasets.get_gencat_graph(
                arg.network_size,
                arg.network_size * 16,
                arg.nb_modules,
                arg.nb_modules * [arg.module_size],
                ctr,
            )
        else:
            G = Datasets.get_scale_free_graph(
                arg.network_size,
                NB_INITIAL_NODES,
                arg.nb_modules,
                arg.module_size,
                P_PROB,
                Q_PROB,
                ctr,
            )
            if arg.removed_edges > 0:
                nbtoremove = int(G.number_of_edges() * arg.removed_edges)
                rng = Random(ctr)
                edges_to_remove = rng.sample(G.edges, nbtoremove)
                G.remove_edges_from(edges_to_remove)
                G.graph["nb_edges"] = G.number_of_edges()

        # initialize the model
        if arg.verbose:
            print("****************")
            print("resultats du run : ",ctr + 1 )
            #model = models.Node2vec()
        if arg.my_model == 'Amine':
            model = models.Node2vec()
            model.init(G)
        elif arg.my_model == 'ManeView':
            model = models.ManeView()
            model.init(G,arg.pvalue,arg.graphe,arg.metrique,arg.my_model,arg.no_singleton,my_dict=Datasets.get_groups(G))
        elif arg.my_model == 'SaeView':
            model = models.SaeView()
            model.init(G,arg.pvalue,arg.graphe,arg.metrique,arg.my_model,arg.no_singleton,my_dict=Datasets.get_groups(G))
        elif arg.my_model == 'Dbscan':
            model = models.DbscanView()
            model.init(G,arg.pvalue,arg.graphe,arg.metrique,arg.my_model,arg.no_singleton,Datasets.get_groups(G),arg.pipe)
        else:
            raise ValueError(f"Modèle inconnu: {arg.my_model}")
       # model.init(G)

        # call module detection method
        start_time = time.perf_counter()
        if not isinstance(model, models.DbscanView):
            module_detection = ModuleDetection(
                    G, model, fitness_fun, background_correction=True
                )
                # do the prediction
            results = module_detection.predict_modules(
                    max_nb_modules=arg.nb_modules, cutoff=0.05
                )
        else:
            results=[]
            cluter=model.dbscan_cluster(eps=0.3,min_samples=3)
            result=model.evaluate_cluter(cluter,Datasets.get_groups(G))
            if result:
              results.append(result[0])
        end_time = time.perf_counter()
        pvalue = 0
        if results:
            pvalue = results[0][2]
        truehits = Datasets.get_groups(G)
        if arg.verbose:
            print("truehits", truehits)
        pred = set()
        nb_clust = 0
        for cl in results:
            if arg.target_module_size:
                if not pred:
                    pred.update(cl[0])
                    continue
                if len(pred) < arg.target_module_size:
                    if abs(arg.target_module_size - len(pred)) > abs(
                        arg.target_module_size - len(pred) - len(cl[0])
                    ):
                        pred.update(cl[0])
                        continue
                    else:
                        break
                else:
                    break
            else:
                nb_clust += 1
                if nb_clust > arg.nb_modules:
                    break
                pred.update(cl[0])

        # merge the content of the true hits in a set
        th = set()
        for key, value in truehits.items():
            th.update(value)

        nb_pred = len(pred)
        nb_th = len(th)
        nb_found = len(pred & th)

        if arg.outfile:
            perf = end_time - start_time
            outfile.write(
                f"{ctr + 1},{perf:.2f},{nb_pred},{nb_th},{nb_found},{pvalue}\n"
            )
            outfile.flush()
        if not arg.verbose:
            continue
        # calculate the F1 score
        res = []
        f1_score = Scores.measure_f1(G, th, pred)
        res.append((len(pred), len(pred & th), f1_score))

        f1_scores.append(f1_score)

        # compute the stats
        quartiles = np.percentile(f1_scores, [25, 50, 75])
        minim = min(f1_scores)
        maxim = max(f1_scores)
        iqr = quartiles[2] - quartiles[0]
        high = min(maxim, quartiles[2] + iqr * 1.5)
        low = max(minim, quartiles[0] - iqr * 1.5)
        out_high = [str(f"{x:.5f}") for x in f1_scores if x > high]
        out_low = [str(f"{x:.5f}") for x in f1_scores if x < low]
        print(
            f"{ctr + 1} len={nb_pred} true_hits={nb_th} found={nb_found} f1_score={f1_scores[-1]:.5f}, mean={statistics.mean(f1_scores):.5f}, variance={statistics.pstdev(f1_scores):.5f}, Q25={quartiles[0]:.5f}, Q50={quartiles[1]:.5f}, Q75={quartiles[2]:.5f}, low={low:.5f}, high={high:.5f}, out_low=[{','.join(out_low)}], out_high=[{','.join(out_high)}] pvalue={pvalue}"
        )
        moyenne=statistics.mean(f1_scores)
        moyenne_formattee = f"{moyenne:.5f}"

    #opreration de sauvergarde et de supression.
        arch=Archiver() 
        if arg.my_model=="Amine":
             arch.save_in_csv("sauvegarde/"+arg.my_model+"/resultats/mon_fichier.csv",
                               {'graphe_run': ctr + 1,'longueur_predit': nb_pred,
                                'true_hits': nb_th,'nombre_noeud_touve': nb_found,
                                'f1_score': f"{f1_scores[-1]:.5f}",
                                'mean': moyenne_formattee ,
                                'Truehits': truehits,'Noeud_predit':pred})
        else:
            arch.save_in_csv("sauvegarde/"+arg.my_model+"/resultats/mon_fichier.csv", 
                             {'graphe_run': ctr + 1,'metrique':arg.metrique, 
                              'modele_vue2':arg.graphe,'longueur_predit': nb_pred,
                              'true_hits': nb_th,'nombre_noeud_touve': nb_found,
                              'f1_score': f"{f1_scores[-1]:.5f}",
                              'mean':moyenne_formattee ,'Truehits': truehits,'Noeud_predit':pred})
        arch.copy_directory(arg.my_model,arch.dest+arg.my_model+str(ctr+1))
        if arg.my_model=="Amine":
            arch.add_to_zip(arch.dest+arg.my_model+".zip",arch.dest+arg.my_model+str(ctr+1))
        else:
            arch.add_to_zip(arch.dest+arg.metrique+"_"+arg.my_model+".zip",arch.dest+arg.my_model+str(ctr+1))
          
        if arg.my_model != "Amine"  :
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/embedding")
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/dataset")
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/graphique_fonction")
        if arg.my_model=="ManeView" :
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/Couples_ids")
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/Couples_nodes")
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/Marches")
            arch.deleteFileInDirectory("sauvegarde/"+arg.my_model+"/Paires")
        arch.delete_directory(repertoire=arch.dest+arg.my_model+str(ctr+1))

    if arg.outfile:
        outfile.close()
