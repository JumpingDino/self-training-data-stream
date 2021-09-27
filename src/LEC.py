from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from utils import normalize
from utils import most_frequent

import numpy as np
import pandas as pd

class Lec:
    """Labelling by Ensemble of clusters

    Args:
        q_nj ([type]): number of labeled data of class j in cluster (Cn)
                       número de dados rotulados de classe j no cluster Cn
        y_r ([type]): Label of class r
                      Rótulos da classe r
        y_Cm ([type]): Label of instances in the cluster Cm
                       Rótulo de instancias no cluster Cm
        z ([type]): Index of labeled clusters
                    Index dos clusteres rotulados
        Q ([type]): Number of clusters
                    Número de clusteres
        t ([type]): Number of classes
                    Número de classes
        Di ([type]): Chunk i
                     Bloco de dados i
        x_u ([type]): Unlabeled instance
                      Dado não rotulado
        K ([type]): Number of clustering algorithms
                    Número de algoritmos de clusterização
        CAp ([type]): Clustering Algorithm p
                      Algoritmo de clusterização p
                      
        DISCLAIMER: Não HA METODOLOGIA CASO HAJA EMPATE NO NUMERO DE ROTULOS
    """
    
    def __init__(self, y_r, Q, Di):
        self.y_r = y_r.copy() #rótulos da classe Di
        self.Di = Di #chunkfor clustering
        self.Q = Q # number of clusters
        
    def kmeans_clustering(self, Di, Q):
        '''Function to do a KMeans clustering defininf a chunk of data and a
        number of clusters'''

        CA = KMeans(n_clusters=Q)
        y_r = CA.fit_predict(Di)
        
        return CA, y_r
            
    
    def label_unlabeled_cluster(self, C_u_element, dist_cluster, C_n_reclass):

        C_u_distances = dist_cluster[C_u_element]

        #create index from clusters
        idx = list(range(len(C_u_distances)))

        #retrieve labels
        labels = []
        for element in idx:
            c = C_n_reclass.get(element)
            if c is None:
                c = np.inf
            labels.append(c)
        labels = np.array(labels)

        #create cluster matrix -> idx (C_n), distances, label
        cluster_matrix = np.c_[np.array(idx),C_u_distances,labels]

        #TO DO: Poderia pegar o segundo cluster com menor distancia
        #filter to don't get the same cluster and get the closest cluster
        fcluster_matrix = cluster_matrix[cluster_matrix[:,1] != 0]
        fcluster_matrix = cluster_matrix[cluster_matrix[:,2] != np.inf]
        close_cluster = fcluster_matrix[fcluster_matrix[:, 1].argsort()][0]

        #get cluster_idx and label
        cluster_idx, cluster_label = close_cluster[0],close_cluster[2]
        
        return cluster_idx, cluster_label

    def unsup_prediction(self, Q, Di, target):

        y_r = target.copy()  
        #### Fit do modelo de cluster e salva em C_n 
        CA, pred_cluster = self.kmeans_clustering(Di, Q)

        C_u = []
        C_n_reclass = {}
        
        for C_n in np.unique(pred_cluster):

            # index dos clusteres
            cluster_insts_idx = np.where(pred_cluster == C_n)[0]

            # rótulos do cluster
            cluster_labels = y_r[cluster_insts_idx]

            #caso cluster haja rótulo, vê classe majoritaria e reclassifica
            if np.sum(cluster_labels >= 0).sum() :
                cntr_labels = Counter(y_r[cluster_insts_idx])
                cluster_mode = most_frequent(cntr_labels,ignore_vals = [-1])
                y_r[cluster_insts_idx] = [cluster_mode for cluster_label in y_r[cluster_insts_idx]]

                #popula dicionario de reclassificaco
                C_n_reclass[C_n] = cluster_mode

            else:
                C_u.append(C_n)
        
        if len(C_u) > 0:
            dist_cluster = pairwise_distances(X = CA.cluster_centers_)

            for C_u_element in C_u:
                cluster_idx, cluster_label = self.label_unlabeled_cluster(C_u_element, dist_cluster, C_n_reclass)

                cluster_insts_idx = np.where(pred_cluster == C_u_element)[0] 
                y_r[cluster_insts_idx] = [cluster_label for z in y_r[cluster_insts_idx]]

        return y_r

    def fit_transform(self, K = 10, to_frame=True):
        results = []
        for _ in range(K):
            pred_round = self.unsup_prediction(Q = self.Q, Di =self.Di, target = self.y_r)
            results.append(pred_round)

        if to_frame:
            results = pd.DataFrame(results).T

        return results


if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris


    def unlabel(x , frac):
        return np.array([-1 if np.random.rand() < frac else label for label in y ])

    X = pd.DataFrame(load_iris()['data'], columns=load_iris()['feature_names'])
    y = load_iris()['target']

    df = pd.concat([X,pd.Series(y,name='class')],axis=1)
    df = df.sample(len(df))

    X = df.drop(columns = 'class').to_numpy()
    Di=X[:,:2]

    y = df['class'].to_numpy()
    y_r = unlabel(y,frac = 0.95)
    y_noisy = y_r.copy()

    ###
    Lec = LEC(y_r,Q=8,Di=Di)
    results = Lec.fit_transform(K=10)
    res_df = pd.DataFrame(results).T
    res_df.columns = [f'label_round_{col}' for col in res_df.columns]
    print(res_df.sample(10))


