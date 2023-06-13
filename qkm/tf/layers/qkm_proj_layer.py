import tensorflow as tf

class QKMProjLayer(tf.keras.layers.Layer):
    """Quantum Kernel Mixture projection layer
    Receives as input a vector and calculates its projection over the 
    layer QKM.
    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, )
    Arguments:
        kernel: a kernel layer
        dim_x: int. the dimension of the input state
        x_train: bool. Whether to train the or not the compoments of the train
                       QKM
        w_train: bool. Whether to train the or not the weights of the compoments 
                       of the train QKM
        n_comp: int. Number of components used to represent 
                 the train QKM
    """

    def __init__(
            self,
            kernel,
            dim_x: int,
            x_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.x_train = x_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.c_w = self.add_weight(
            "c_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 

    def call(self, inputs):
        comp_w = tf.abs(self.c_w) + 1e-10
        # normalize comp_w to sum to 1
        comp_w = comp_w / tf.reduce_sum(comp_w)
        in_v = inputs[:, tf.newaxis, :]
        out_vw = self.kernel(in_v, self.c_x) ** 2 # shape (b, 1, n_comp)
        out_w = tf.einsum('...j,...ij->...', comp_w, out_vw, optimize="optimal")
        return out_w

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "w_train": self.w_train,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)