"""
Various methods to cluster embeddings.
"""

from distinctipy import distinctipy
import matplotlib as mpl
import numpy as np
import os
import scipy.linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import spectral as spy


class Cluster:
    """ Represents a clustering layer of single hyperspectral image.

    Parameters
    ----------
    cube : HyperCube
        HyperCube with `emb` to cluster.

    Attributes
    ----------
    cube : HyperCube
        See above.
    clus : (A,B) ndarray
        -1 for unclustered areas and a consecutive natural number for each other clustering.
    """

    def __init__(self, cube):
        self.cube = cube
        self.clus = np.full(cube.cube.shape[:2], -1, dtype=np.int16)

    def k_means(self, n_clusters):
        """ Performs k-means clustering on embedding. 

        Parameters
        ----------
        n_clusters : int
            Number of clusters to produce.

        """

        self.clus[self.cube.mask] = KMeans(n_clusters=n_clusters).fit(
            self.cube.emb[self.cube.mask]).labels_

    def gaussian_mixture(self, n_clusters):
        """ Performs gaussian mixture model clustering on embedding. 

        Parameters
        ----------
        n_clusters : int
            Number of clusters to produce.

        """

        self.clus[self.cube.mask] = GaussianMixture(
            n_components=n_clusters).fit_predict(self.cube.emb[self.cube.mask])

    # TODO: Make safe
    def hierarchical_gaussian_mixture(self, depth):
        """ Performs hierarchical gaussian mixture model clustering on
        embedding by using a series of two-cluster gaussian mixture models.

        Parameters
        ----------
        depth : int
            2**depth clusters will be produced.

        """

        def _hierarchical_gaussian_mixture(emb, depth):
            clus = GaussianMixture(
                n_components=2, random_state=0).fit_predict(emb)
            if depth:
                i_l = clus == 0
                i_r = clus == 1
                clus[i_l] = _hierarchical_gaussian_mixture(emb[i_l], depth-1)
                clus[i_r] = _hierarchical_gaussian_mixture(
                    emb[i_r], depth-1)+np.max(clus)+1

            return clus

        self.clus[self.cube.mask] = _hierarchical_gaussian_mixture(
            self.cube.emb[self.cube.mask], depth)

    def combine_spectrally_similar(self, n_clusters):
        """ Iteratively combines clusters with the smallest spectral angle
        between their mean pixels.

        Parameters
        ----------
        n_clusters : int
            Stopping condition for when to stop combining clusters.

        """

        
        X = self.cube.cube[self.cube.mask]
        clus = self.clus[self.cube.mask]
        orig_num_clus= len(np.unique(clus))
        angs = Cluster.get_angs(self.clus, self.cube.cube, orig_num_clus)

        
        new_angs=angs

        #####################################33
        cnt = 0
        while (len(np.unique(clus))) > n_clusters:        

          print("reduction number ",cnt,'...')
          
          new, old = np.unravel_index(np.argmin(new_angs), new_angs.shape)
          print("combining cluster",old , "into cluster ", new)
          

          clus[clus == old] = new
          print("new cluster labels", np.unique(clus))

          new_angs=Cluster.get_angs(clus, X,orig_num_clus)
          #print("new angle matrix after reduction:, ",new_angs)
          cnt+=1

        print("reduction loop is done, the remaining clusters are:", np.unique(clus) )
        print("the number of reduced clusters is, " ,
              len(np.unique(clus)),"and should equal, ", n_clusters )

        self.clus[self.cube.mask] = clus
        #######################################3333


    def save_clustering(self, file_path, color=False):
        """ Save clustering image to `file_path`.

        Parameters
        ----------
        file_path : PathLike
            Output path for `clus`.
        color : bool
            Whether to apply a false coloring to the clustering.
            WISER is not currently able to do so, so this is helpful.

        """

        clus = self.clus
        if color:
            colors = ['#000000'] + distinctipy.get_colors(2+np.max(self.clus))
            clus = mpl.colors.ListedColormap(colors)(self.clus+1)

        if os.path.splitext(file_path)[1] == '.npy':
            np.save(file_path, clus)
        elif os.path.splitext(file_path)[1] == '.npz':
            np.savez(file_path, clus)
        else:
            spy.envi.save_classification(file_path, clus, force=True)

    @staticmethod
    def get_ang(u, v):
        """ Computes the spectral angle between two vectors. 

        Parameters
        ----------
        u : (A,) ndarray
            Mean pixel of first cluster.
        v : (A,) ndarray
            Mean pixel of second cluster.

        Returns
        -------
        theta : float
            Angle between `u` and `v`.

        """
        param1=scipy.linalg.norm(u)
        param2=scipy.linalg.norm(v)
        ans=np.arccos(np.dot(u, v)/(1e-8+param1*param2))


        return ans
    # Requires clus with classes 0...n
    @staticmethod
    def get_angs(clus, X, orig_n_clus):
        """ Computes lower triangular matrix of spectral angle between every
        pairing of clusters. 
        *** should this be the upper triangular matrix?**

        Parameters
        ----------
        clus : (A,) ndarray
            Clustering map of classes 0...n.
        X : (A, B) ndarray
            Corresponding spectral data for clustering map.

        Returns
        -------
        angs : (MAX(clus)+1, MAX(clus)+1) ndarray
            Lower triangular matrix holding spectral angle between every
            pairing of clusters.

        """

        #n_clusters = np.max(clus)+1
        n_clusters = orig_n_clus
        all_clusters=np.unique(clus)[1:]
        print(all_clusters)
        means=[]

        
        for i in all_clusters:
          means_ele = np.array(np.mean(X[clus == i], axis=0))
          means.append(means_ele)
          
        means=np.array(means)
        angs = np.full((n_clusters, n_clusters), np.pi, dtype=np.float64)
        #print("means is ", means.shape)
        #print("exaample means[i]", means[0], "and means[j] ", means[0])

        if n_clusters==len(all_clusters):
          for i in range(n_clusters):
              u = means[i]
              for j in range(i):
                  v = means[j]
                  #print("u input to get_ang is ",u.shape)
                  #print("v input to get_ang is ",v.shape)
                  
                  angs[j, i] = Cluster.get_ang(u, v)
        else:
          for i in range(len(all_clusters)):# [1,2,4,5] [0,1,2,3,4,5]
            u = means[i]
            for j in range(i):
              v=means[j]
              #print([j])
              #print([i])
              #print()
              
              angs[int(all_clusters[j]),int(all_clusters[i])]= Cluster.get_ang(u,v)



        return angs
