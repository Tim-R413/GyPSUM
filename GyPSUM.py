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
    npy_20_img='/content/drive/MyDrive/Covercrop_segmentation/CC_datasets/hyperspec/soil_mask_dataset/combinedSM_raster_20ran.npy'
    envi_img_ex = '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.dat'
    npy_86_img='/content/drive/MyDrive/Covercrop_segmentation/CC_datasets/hyperspec/combined_arrays/all_images.npy'





    start = time.time(), 
    cube = HyperCube(img_path= '/content/drive/MyDrive/Covercrop_segmentation/CC_datasets/hyperspec/soil_mask_dataset/combinedSM_raster_20ran.npy' , hdr_path= '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.hdr',band_wv='/content/GyPSUM/Band_wavelengths.npy', img_type='npy')
    

    print('checking first pixel:',np.sum(cube.cube[0,0]))
    cube.unmask_value()
    
    cube.clip()
    #cube.ratio('40ffratiospec.npy')
    #cube.spectral_subset(1050, 2550)
    
    cube.normalize()
    #cube.display_img()
    # cube.remove_continuum()
    cube.standardize()
    #cube.set_n_components(6)

    print('checking first pixel before feature extraction:',np.sum(cube.cube[0,0]))
    cube.hysime()
    cube.autoencoder(epochs=3)
    #print(cube.cube.shape)
    # cube.pca()

    
    number_clusters= 5
    print('number of components found:',cube.n_components)
    print('number of clusters to produce will be:', number_clusters)
    
    print('check first pixel before clustering',np.sum(cube.cube[0,0]))
    clus = Cluster(cube)
    # clus.k_means(5)
    clus.gaussian_mixture(2*cube.n_components)
    # clus.hierarchical_gaussian_mixture(4)
    clus.combine_spectrally_similar(number_clusters)


    print("final list of unique cluster names",np.unique(clus.clus))
    

    end = time.time()
   
    print('The pipeline took', end-start[0], 'seconds to complete.')

    cube.save_cube('cube.npy')
    cube.save_mask('mask.npy')
    cube.save_emb('emb.npy')
    clus.save_clustering('clustering_img.npy', color=True)
    np.save('clustering_arr',clus.clus )


if __name__ == '__main__':
    app.run(main)
