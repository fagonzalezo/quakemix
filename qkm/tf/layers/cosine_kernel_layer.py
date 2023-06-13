import tensorflow as tf

class CosineKernelLayer(tf.keras.layers.Layer):
    def __init__(self):
        '''
        Builds a layer that calculates the cosine kernel between two set of vectors
        '''
        super(CosineKernelLayer, self).__init__()
        self.eps = 1e-6

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A = tf.math.divide_no_nan(A, 
                                  tf.expand_dims(tf.norm(A, axis=-1), axis=-1))
        B = tf.math.divide_no_nan(B, 
                                  tf.expand_dims(tf.norm(B, axis=-1), axis=-1))
        K = tf.einsum("...nd,md->...nm", A, B)
        return K
    
    def log_weight(self):
        return 0
