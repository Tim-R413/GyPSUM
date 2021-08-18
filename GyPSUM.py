"""
Demo of the complete GyPSUM pipeline. Options appear in commented out lines.
"""

from absl import app
# from absl import flags
import os
import time

from feature_extraction import HyperCube
from cluster import Cluster
import numpy as np


def main(argv):
    # Removes TensorFlow debugging output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start = time.time(), 
    cube = HyperCube(img_path= '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.dat', hdr_path= '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.hdr',band_wv='/content/GyPSUM/Band_wavelengths.npy')
    
    print('check first pixel',np.sum(cube.cube[0,0]))
    cube.unmask_value()
    
    cube.clip()
    #cube.ratio('40ffratiospec.npy')
    #cube.spectral_subset(1050, 2550)
    
    cube.normalize()
    #cube.display_img()
    # cube.remove_continuum()
    cube.standardize()
    #cube.set_n_components(6)

    print('check first pixel before feature extraction',np.sum(cube.cube[0,0]))
    cube.hysime()
    cube.autoencoder(epochs=3)
    #print(cube.cube.shape)
    # cube.pca()

    
    number_clusters= cube.n_components
    print('number of clusters to produce will be:',number_clusters )
    
    print('check first pixel before clustering',np.sum(cube.cube[0,0]))
    clus = Cluster(cube)
    # clus.k_means(5)
    clus.gaussian_mixture(6)#number_clusters)
    # clus.hierarchical_gaussian_mixture(4)
    #clus.combine_spectrally_similar(6)
    print(np.unique(clus.clus))
    print(np.max(clus.clus))

    end = time.time()
   
    print('The pipeline took', end-start[0], 'seconds to complete.')

    cube.save_cube('cube.npy')
    cube.save_mask('mask.npy')
    cube.save_emb('emb.npy')
    clus.save_clustering('clustering.npy', color=True)


if __name__ == '__main__':
    app.run(main)
