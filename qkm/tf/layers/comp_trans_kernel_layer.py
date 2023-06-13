import tensorflow as tf

class CompTransKernelLayer(tf.keras.layers.Layer):
    def __init__(self, transform, kernel):
        '''
        Composes a transformation and a kernel to create a new
        kernel.
        Arguments:
            transform: a function f that transform the input before feeding it to the 
                    kernel
                    f:(bs, d) -> (bs, D) 
            kernel: a kernel function
                    k:(bs, n, D)x(m, D) -> (bs, n, m)
        '''
        super(CompTransKernelLayer, self).__init__()
        self.transform = transform
        self.kernel = kernel

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape = tf.shape(A) # (bs, n, d)
        A = tf.reshape(A, [shape[0] * shape[1], shape[2]])
        A = self.transform(A)
        dim_out = tf.shape(A)[1]
        A = tf.reshape(A, [shape[0], shape[1], dim_out])
        B = self.transform(B)
        return self.kernel(A, B)
    
    def log_weight(self):
        return self.kernel.log_weight()