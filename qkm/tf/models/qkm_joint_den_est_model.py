import tensorflow as tf
from ..layers import RBFKernelLayer, QKMProjLayer, \
                     CosineKernelLayer, CrossProductKernelLayer
import numpy as np
from sklearn.metrics import pairwise_distances


class QKMJointDenEstModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 sigma,
                 n_comp,
                 trainable_sigma=True,
                 min_sigma=1e-3):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.kernel_x = RBFKernelLayer(sigma, dim=dim_x, 
                                       trainable=trainable_sigma,
                                       min_sigma=min_sigma)
        self.kernel_y = CosineKernelLayer()
        self.kernel = CrossProductKernelLayer(dim1=dim_x, kernel1=self.kernel_x, kernel2=self.kernel_y)
        self.qkmproj = QKMProjLayer(self.kernel,
                                dim_x=dim_x + dim_y,
                                n_comp=n_comp)

    def call(self, inputs):
        log_probs = (tf.math.log(self.qkmproj(inputs) + 1e-12)
                     + self.kernel.log_weight())
        self.add_loss(-tf.reduce_mean(log_probs))
        return log_probs
    
    def init_components(self, samples_xy,
                        sigma):
        self.kernel_x.sigma.assign(sigma)
        self.qkmproj.c_x.assign(samples_xy)
        self.qkmproj.c_w.assign(tf.ones((self.n_comp,)) / self.n_comp)
