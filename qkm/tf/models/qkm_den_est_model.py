import tensorflow as tf
from ..layers import RBFKernelLayer, QKMProjLayer
import numpy as np
from sklearn.metrics import pairwise_distances

class KQMDenEstModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 sigma,
                 n_comp):
        super().__init__()
        self.dim_x = dim_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma, dim=dim_x)
        self.qkmproj = QKMProjLayer(self.kernel,
                                dim_x=dim_x,
                                n_comp=n_comp)

    def call(self, inputs):
        log_probs = (tf.math.log(self.qkmproj(inputs) + 1e-12)
                     + self.kernel.log_weight())
        self.add_loss(-tf.reduce_mean(log_probs))
        return log_probs
    
    def init_components(self, samples_x, init_sigma=False, sigma_mult=1):
        if init_sigma:
            distances = pairwise_distances(samples_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.qkmproj.c_x.assign(samples_x)
        self.qkmproj.c_w.assign(tf.ones((self.n_comp,)) / self.n_comp)
