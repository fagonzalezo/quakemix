import tensorflow as tf
import numpy as np

class RBFKernelLayer(tf.keras.layers.Layer):
    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3):
        '''
        Builds a layer that calculates the rbf kernel between two set of vectors
        Arguments:
            sigma: RBF scale parameter. If it is a tf.Variable it will be used as is.
                     Otherwise it will create a trainable variable with the given value.
        '''
        super().__init__()
        #super(RBFKernelLayer, self).__init__()
        if type(sigma) is tf.Variable:
            self.sigma = sigma
        else:
            self.sigma = tf.Variable(sigma, dtype=tf.float32, trainable=trainable)
        self.dim = dim
        self.min_sigma = min_sigma

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape_A = tf.shape(A)
        shape_B = tf.shape(B) 
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2 # shape (bs, n, 1)
        B_norm = tf.norm(B, axis=-1)[tf.newaxis, tf.newaxis, :] ** 2 # shape (1, 1, m)
        A_reshaped = tf.reshape(A, [-1, shape_A[2]]) # shape (bs * n, d)
        AB = tf.matmul(A_reshaped, B, transpose_b=True) # shape (bs * n, m)
        AB = tf.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]]) # shape (bs, n, m)
        dist2 = A_norm + B_norm - 2. * AB # shape (bs, n, m)
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        sigma = tf.clip_by_value(self.sigma, self.min_sigma, np.inf)
        K = tf.exp(-dist2 / (2. * sigma ** 2.)) # type: ignore
        return K
    
    def log_weight(self):
        sigma = tf.clip_by_value(self.sigma, self.min_sigma, np.inf)
        return - self.dim * tf.math.log(sigma + 1e-12) - self.dim * np.log(4 * np.pi)  

class MemRBFKernelLayer(RBFKernelLayer):
    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3):
        '''
        Builds a layer that calculates the rbf kernel between two set of vectors
        Arguments:
            sigma: RBF scale parameter. If it is a tf.Variable it will be used as is.
                     Otherwise it will create a trainable variable with the given value.
        '''
        super().__init__(sigma, dim, trainable, min_sigma)

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (bs, m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape_A = tf.shape(A)
        shape_B = tf.shape(B)
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2 # shape (bs, n, 1)
        B_norm = tf.norm(B, axis=-1)[:, tf.newaxis, :] ** 2 # shape (bs, 1, m)
        # A_reshaped = tf.reshape(A, [-1, shape_A[2]])
        AB = tf.matmul(A, B, transpose_b=True) # shape (bs, n, m) 
        dist2 = A_norm + B_norm - 2. * AB # shape (bs, n, m)
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        sigma = tf.clip_by_value(self.sigma, self.min_sigma, np.inf)
        K = tf.exp(-dist2 / (2. * sigma ** 2.)) # type: ignore
        return K
    
