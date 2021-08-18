"""
Demo of the complete GyPSUM pipeline. Options appear in commented out lines.
"""

from absl import app
# from absl import flags
import os
import time

from feature_extraction import HyperCube
from cluster import Cluster


def main(argv):
    # Removes TensorFlow debugging output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start = time.time(), 
    cube = HyperCube(img_path= '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.dat', hdr_path= '/content/GyPSUM/ENVI_images/P306_1_1B4C1T4V.hdr',band_wv='')
    
  
    cube.unmask_value()
    
    cube.clip()
    #cube.ratio('40ffratiospec.npy')
    #cube.spectral_subset(1050, 2550)
    
    cube.normalize()
    #cube.display_img()
    # cube.remove_continuum()
    cube.standardize()
    #cube.set_n_components(20)

    cube.hysime()
    cube.autoencoder(epochs=3)
    # cube.pca()

    clus = Cluster(cube)
    # clus.k_means(5)
    clus.gaussian_mixture(15)
    # clus.hierarchical_gaussian_mixture(4)
    # clus.combine_spectrally_similar(15)

    end = time.time()
    print(f'The pipeline took {end-start} seconds to complete.')

    cube.save_cube('cube.hdr')
    cube.save_mask('mask.hdr')
    cube.save_emb('emb.hdr')
    clus.save_clustering('clustering.hdr', color=True)


if __name__ == '__main__':
    app.run(main)
