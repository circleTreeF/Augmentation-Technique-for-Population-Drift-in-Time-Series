import numpy as np



def save_density_with_bandwidth(out_path, density, bandwidth):
    save_path = out_path + str(bandwidth) + '.npy'
    np.save(save_path, density)
