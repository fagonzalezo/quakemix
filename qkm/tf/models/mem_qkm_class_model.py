import tensorflow as tf
import faiss
import numpy as np
from ..layers import MemRBFKernelLayer, MemQKMLayer
from ..utils import pure2dm, dm2discrete
from sklearn.metrics import pairwise_distances

class MemQKMClassModel(tf.keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp,
                 index = None, 
                 sigma=0.1):
        super().__init__() 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.index = index
        self.kernel = MemRBFKernelLayer(sigma=sigma, 
                                         dim=encoded_size, 
                                         trainable=True)
        self.mqkm = MemQKMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp)
    def call(self, input, training=False):
        x_enc, neighbors = input
        x_neigh = tf.gather(self.samples_x, neighbors, axis=0)
        y_neigh = tf.gather(self.samples_y, neighbors, axis=0)
        rho_y = self.mqkm([x_enc, x_neigh, y_neigh])
        probs = dm2discrete(rho_y)
        return probs

    def create_index(self, samples_x, samples_y):
        self.samples_x = self.encoder(samples_x)
        self.samples_y = tf.constant(samples_y, tf.float32)
        self.index = faiss.IndexFlatL2(self.encoded_size)
        self.index.add(self.samples_x)

    def create_train_ds(self, batch_size):
        def tf_search(x, y):
            def search(x):
                _, I = self.index.search(x, self.n_comp + 1)
                return I[:, 1:]
            return (x, tf.numpy_function(search, [x],
                (tf.int64))), y
        ds = tf.data.Dataset.from_tensor_slices(
            (self.samples_x, self.samples_y))
        ds = ds.shuffle(10000).batch(batch_size)
        ds = (ds.
                map(tf_search).
                cache().
                prefetch(tf.data.experimental.AUTOTUNE))
        return ds
    
    def create_test_ds(self, test_x, test_y, batch_size):
        def tf_search(x, y):
            def search(x):
                _, I = self.index.search(x, self.n_comp)
                return I
            x_enc = self.encoder(x)
            return (x_enc, tf.numpy_function(search, [x_enc],
                (tf.int64))), y
        ds = tf.data.Dataset.from_tensor_slices(
            (test_x, test_y))
        ds = ds.batch(batch_size)
        ds = (ds.
                map(tf_search).
                cache().
                prefetch(tf.data.experimental.AUTOTUNE))
        return ds
    
    def create_predict_ds(self, test_x):
        def tf_search(x):
            def search(x):
                _, I = self.index.search(x, self.n_comp)
                return I
            x_enc = self.encoder(x)
            return x_enc, tf.numpy_function(search, [x_enc],
                (tf.int64))
        ds = tf.data.Dataset.from_tensor_slices(
            test_x).batch(test_x.shape[0])
        ds = (ds.
                map(tf_search).
                cache().
                prefetch(tf.data.experimental.AUTOTUNE))
        return next(iter(ds))
    
