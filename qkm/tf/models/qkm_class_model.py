import tensorflow as tf
import numpy as np
from ..layers import QKMLayer, RBFKernelLayer
from ..utils import pure2dm, dm2discrete
from sklearn.metrics import pairwise_distances

class QKMClassModel(tf.keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp, 
                 sigma=0.1):
        super().__init__() 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma=sigma, 
                                         dim=encoded_size, 
                                         trainable=True)
        self.qkm = QKMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp)
    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y =self.qkm(rho_x)
        probs = dm2discrete(rho_y)
        return probs

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        encoded_x = self.encoder(samples_x)
        if init_sigma:
            distances = pairwise_distances(encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.qkm.c_x.assign(encoded_x)
        self.qkm.c_y.assign(samples_y)
        self.qkm.c_w.assign(tf.ones((self.n_comp,)) / self.n_comp)

