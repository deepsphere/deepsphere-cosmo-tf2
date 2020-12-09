from .gnn_layers import *


class HealpyPool(Model):
    """
    A pooling layer for healy maps, makes use of the fact that a pixels is always divided into 4 subpixels when
    increasing the nside of a HealPix map
    """

    def __init__(self, p, pool_type="MAX", **kwargs):
        """
        initializes the layer
        :param p: reduction factor >=1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param pool_type: type of pooling, can be "MAX" or  "AVG"
        :param kwargs: additional kwargs passed to the keras pooling layer
        """
        # This is necessary for every Layer
        super(HealpyPool, self).__init__(name='')

        # check p
        if not p >= 1:
            raise IOError("The reduction factors has to be at least 2!")

        # save variables
        self.p = p
        self.filter_size = int(4**p)
        self.pool_type = pool_type
        self.kwargs = kwargs

        if pool_type == "MAX":
            self.filter = tf.keras.layers.MaxPool1D(pool_size=self.filter_size, strides=self.filter_size,
                                                    padding='valid', data_format='channels_last', **kwargs)
        elif pool_type == "AVG":
            self.filter =  tf.keras.layers.AveragePooling1D(pool_size=self.filter_size, strides=self.filter_size,
                                                            padding='valid', data_format='channels_last', **kwargs)
        else:
            raise IOError(f"Pooling type not understood: {self.pool_type}")

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        n_nodes = int(input_shape[1])
        if n_nodes % self.filter_size != 0:
            raise IOError("Input shape {input_shape} not compatible with the filter size {self.filter_size}")

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        return self.filter(input_tensor)


class HealpyPseudoConv(Model):
    """
    A pseudo convolutional layer on Healpy maps. It makes use of the Healpy pixel scheme and reduces the nside by
    averaging the pixels into bigger pixels using learnable weights
    """

    def __init__(self, p, Fout, kernel_initializer=None, **kwargs):
        """
        initializes the layer
        :param p: reduction factor >=1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param Fout: number of output channels
        :param kernel_initializer: initializer for kernel init
        :param kwargs: additional keyword arguments passed to the kreas 1D conv layer
        """
        # This is necessary for every Layer
        super(HealpyPseudoConv, self).__init__(name='')

        # check p
        if not p >= 1:
            raise IOError("The reduction factors has to be at least 1!")

        # save variables
        self.p = p
        self.filter_size = int(4 ** p)
        self.Fout = Fout
        self.kernel_initializer = kernel_initializer
        self.kwargs = kwargs

        # create the files
        self.filter = tf.keras.layers.Conv1D(self.Fout, self.filter_size, strides=self.filter_size,
                                             padding='valid', data_format='channels_last',
                                             kernel_initializer=self.kernel_initializer, **self.kwargs)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        n_nodes = int(input_shape[1])
        if n_nodes % self.filter_size != 0:
            raise IOError(f"Input shape {input_shape} not compatible with the filter size {self.filter_size}")
        self.filter.build(input_shape)

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        return self.filter(input_tensor)


class HealpyPseudoConv_Transpose(Model):
    """
    A pseudo transpose convolutional layer on Healpy maps. It makes use of the Healpy pixel scheme and increases
    the nside by applying a transpose convolution to the pixels into bigger pixels using learnable weights
    """

    def __init__(self, p, Fout, kernel_initializer=None, **kwargs):
        """
        initializes the layer
        :param p: Boost factor >=1 of the nside -> number of nodes increases by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param Fout: number of output channels
        :param kernel_initializer: initializer for kernel init
        :param kwargs: additional keyword arguments passed to the keras transpose conv layer
        """
        # This is necessary for every Layer
        super(HealpyPseudoConv_Transpose, self).__init__(name='')

        # check p
        if not p >= 1:
            raise IOError("The boost factors has to be at least 1!")

        # save variables
        self.p = p
        self.filter_size = int(4 ** p)
        self.Fout = Fout
        self.kernel_initializer = kernel_initializer
        self.kwargs = kwargs

        # create the files
        self.filter = tf.keras.layers.Conv2DTranspose(self.Fout, (1, self.filter_size), strides=(1, self.filter_size),
                                                      padding='valid', data_format='channels_last',
                                                      kernel_initializer=self.kernel_initializer, **self.kwargs)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        input_shape = list(input_shape)
        n_nodes = input_shape[1]
        if n_nodes % self.filter_size != 0:
            raise IOError(f"Input shape {input_shape} not compatible with the filter size {self.filter_size}")

        # add the additional dim
        input_shape.insert(1, 1)

        self.filter.build(input_shape)

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        input_tensor = tf.expand_dims(input_tensor, axis=1)
        return tf.squeeze(self.filter(input_tensor), axis=1)


class HealpyChebyshev():
    """
    A helper class for a Chebyshev5 layer using healpy indices instead of the general Layer
    """
    def __init__(self, K, Fout=None, initializer=None, activation=None, use_bias=False,
                 use_bn=False, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kwargs: additional keyword arguments passed on to add_weight
        """
        # we only save the variables here
        self.K = K
        self.Fout = Fout
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kwargs = kwargs

    def _get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: Chebyshev5 layer that can be called
        """

        # now we init the layer
        return Chebyshev(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                          use_bias=self.use_bias, use_bn=self.use_bn, **self.kwargs)


class HealpyMonomial():
    """
    A graph convolutional layer using Monomials
    """
    def __init__(self, K, Fout=None, initializer=None, activation=None, use_bias=False, use_bn=False, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # we only save the variables here
        self.K = K
        self.Fout = Fout
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kwargs = kwargs

    def _get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: Monomial layer that can be called
        """

        # now we init the layer
        return Monomial(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                        use_bias=self.use_bias, use_bn=self.use_bn, **self.kwargs)


class Healpy_ResidualLayer():
    """
    A generic residual layer of the form
    in -> layer -> layer -> out + in
    with optional batchnorm in the end
    """

    def __init__(self, layer_type, layer_kwargs, activation=None, act_before=False, use_bn=False,
                 norm_type="batch_norm", bn_kwargs=None, alpha=1.0):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, either "CHEBY" or "MONO" for chebychev or monomials
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batchnorm inbetween the layers
        :param norm_type: type of batch norm, either batch_norm for normal batch norm, layer_norm for
                          tf.keras.layers.LayerNormalization or moving_norm for special_layer.MovingBatchNorm
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        """

        # we only save the variables here
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        self.activation = activation
        self.act_before = act_before
        self.use_bn = use_bn
        self.norm_type = norm_type
        self.bn_kwargs = bn_kwargs
        self.alpha = alpha

    def _get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: GCNN_ResidualLayer layer that can be called
        """
        # we add the graph laplacian to all kwargs
        self.layer_kwargs.update({"L": L})

        return GCNN_ResidualLayer(layer_type=self.layer_type, layer_kwargs=self.layer_kwargs,
                                  activation=self.activation, act_before=self.act_before,
                                  use_bn=self.use_bn, norm_type=self.norm_type,
                                  bn_kwargs=self.bn_kwargs, alpha=self.alpha)
