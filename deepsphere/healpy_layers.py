from .gnn_layers import *
from .gnn_transformers import *

import os
import healpy as hp
from sklearn.neighbors import BallTree
from typing import Union, Optional
from tqdm import tqdm
from . import utils
np.set_printoptions(precision=1)

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
        super(HealpyPool, self).__init__()

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

    def call(self, input_tensor):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
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
        super(HealpyPseudoConv, self).__init__()

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

    def call(self, input_tensor):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
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
        super(HealpyPseudoConv_Transpose, self).__init__()

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

    def call(self, input_tensor):
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

    def _get_layer(self, L, n_matmul_splits=1):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the 
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :return: Chebyshev5 layer that can be called
        """

        # now we init the layer
        return Chebyshev(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                          use_bias=self.use_bias, use_bn=self.use_bn, n_matmul_splits=n_matmul_splits, **self.kwargs)


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

    def _get_layer(self, L, n_matmul_splits=1):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the 
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :return: Monomial layer that can be called
        """

        # now we init the layer
        return Monomial(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                        use_bias=self.use_bias, use_bn=self.use_bn, n_matmul_splits=n_matmul_splits, **self.kwargs)


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

    def _get_layer(self, L, n_matmul_splits=1):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the 
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :return: GCNN_ResidualLayer layer that can be called
        """
        # we add the graph laplacian to all kwargs
        self.layer_kwargs.update({"L": L})
        self.layer_kwargs.update({"n_matmul_splits": n_matmul_splits})

        return GCNN_ResidualLayer(layer_type=self.layer_type, layer_kwargs=self.layer_kwargs,
                                  activation=self.activation, act_before=self.act_before,
                                  use_bn=self.use_bn, norm_type=self.norm_type,
                                  bn_kwargs=self.bn_kwargs, alpha=self.alpha)

class Healpy_ViT(Graph_ViT):
    """
    This is a wrapper for the Graph_ViT to have everything consistent syntax between everything
    Since this layer does not need any additional quantities like the graph laplacian that is only available
    at runtime, it is literally the same as Graph_ViT
    """

    def __init__(self, p, key_dim, num_heads, positional_encoding=True, n_layers=1, activation="relu",
                 layer_norm=True):
        """
        Creates a visual transformer according to:
        https://arxiv.org/pdf/2010.11929.pdf
        by dividing the healpy graph into super pixels
        :param p: reduction factor >1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param key_dim: Dimension of the key, query and value for the embedding in the multi head attention for each
                        head. Note that this means that the initial embedding will be key_dim*num_heads
        :param num_heads: Number of heads to learn in the multi head attention
        :param positional_encoding: If True, add positional encoding to the superpixel embedding in the beginning.
        :param n_layers: Number of TransformerEncoding layers after the initial embedding
        :param activation: The activation function to use for the multiheaded attention
        :param layer_norm: If layernorm should be used for the multiheaded attention
        """

        # just do the super init
        super(Healpy_ViT, self).__init__(p=p, key_dim=key_dim, num_heads=num_heads,
                                         positional_encoding=positional_encoding, n_layers=n_layers,
                                         activation=activation, layer_norm=layer_norm)

class Healpy_Transformer():
    """
    The wrapper layer for the Graph_Transformer layer
    """

    def __init__(self, key_dim, num_heads, positional_encoding=True, n_layers=1, activation="relu",
                 layer_norm=True):
        """
        Creates a visual transformer according to:
        https://arxiv.org/pdf/2010.11929.pdf
        by dividing the healpy graph into super pixels
        :param key_dim: Dimension of the key, query and value for the embedding in the multi head attention for each
                        head. Note that this means that the initial embedding will be key_dim*num_heads
        :param num_heads: Number of heads to learn in the multi head attention
        :param positional_encoding: If True, add positional encoding to the superpixel embedding in the beginning.
        :param n_layers: Number of TransformerEncoding layers after the initial embedding
        :param activation: The activation function to use for the multiheaded attention
        :param layer_norm: If layernorm should be used for the multiheaded attention
        """

        # save variables
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.n_layers = n_layers
        self.activation = activation
        self.layer_norm = layer_norm

    def _get_layer(self, A):
        """
        initializes the actual layer, should be called once the graph adjacency matrix has been calculated
        :param A: the graph Adjacency matrix
        :return: Graph_Transformer layer that can be called
        """

        return Graph_Transformer(A=A, key_dim=self.key_dim, num_heads=self.num_heads,
                                 positional_encoding=self.positional_encoding, n_layers=self.n_layers,
                                 activation=self.activation, layer_norm=self.layer_norm)

class HealpyBernstein():
    """
    A helper class for a Bernstein layer using healpy indices instead of the general Layer
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

    def _get_layer(self, L, n_matmul_splits=1):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the 
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :return: Chebyshev5 layer that can be called
        """

        # now we init the layer
        return Bernstein(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                          use_bias=self.use_bias, use_bn=self.use_bn, n_matmul_splits=n_matmul_splits, **self.kwargs)

class HealpySmoothing(Model):
    """
    A layer that smoothes a Healpix map with a Gaussian kernel.
    """

    def __init__(
        self,
        # pixels
        nside: int,
        indices: np.ndarray,
        nest: bool = True,
        mask: Optional[tf.Tensor] = None,
        # smoothing
        fwhm: Optional[Union[int, float, list]] = None,
        sigma: Optional[Union[int, float, list]] = None,
        n_sigma_support: Union[int, float] = 3,
        arcmin: bool = True,
        per_channel_repetitions: Optional[Union[list, np.ndarray]] = None,
        # computational
        data_path: Optional[str] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        """Create the sparse kernel tensor with which the maps are smoothed.

        Note that the smoothing is always done with a single base sigma. When different smoothing scales are specified
        for the different input channels,that kernel is applied repeatedly to channels which require a larger smoothing
        scale, by exploiting the fact that the convolution of two Gaussians with standard deviations sigma_1 and
        sigma_2 is a Gaussian with sigma_3 = sqrt(sigma_1^2 + sigma_2^2). This implementation saves GPU memory, as the
        sparse kernel matrix can grow to be very large.

        Args:
            nside (int): The healpy nside of the input.
            indices (np.ndarray): 1d array of indices, corresponding to the pixel ids of the input map footprint.
            nest (bool, optional): Whether the maps are stored in healpix NEST ordering. Defaults to True, which is
                always the case for DeepSphere networks.
            mask (Optional[tf.Tensor], optional): Boolean tensor of shape (n_indices, 1) or (n_indices, n_channels)
                that indicates which part of the patch defined by the indices is actually populated. Defaults to None,
                then no additional masking is applied and the maps bleed into the zero padding.
            fwhm (Optional[Union[int, float]], optional): FWHM of the Gaussian smoothing kernel. Can be either a single
                or per channel number. In the latter case, the smoothing scale of the kernel is chosen as the smallest
                value and the rest achieved by smoothing repeatedly. Defaults to None, then sigma needs to be
                specified.
            sigma (Optional[Union[int, float]], optional): Identical functionality as the fwhm argument, but specifies
                the standard deviation of the Gaussian smoothing kernel instead. Defaults to None, then fwhm needs to
                be specified.
            n_sigma_support (Union[int, float], optional): Determines the radius from which the smoothing is
                calculated. Specifically, this value determines which nearest neighbors are included. Defaults to 3,
                then roughly 99.7% of the Gaussian probability mass is accounted for.
            arcmin (bool, optional): Whether fwhm and sigma are specified in arcmin or radian. Defaults to True.
            per_channel_repetitions (Optional[Union[list, np.ndarray]], optional): When a single value is specified for
                fwhm or sigma, this argument determines the per channel number of times the smoothing kernel is
                applied. Defaults to None.
            data_path (Optional[str], optional): Path where the sparse kernel tensor is stored to, and if available,
                loaded from. Defaults to None, then the sparse kernel tensor is neither saved nor loaded.
            max_batch_size (int, optional): Maximal batch size this network is supposed to handle. This determines the
                number of splits in the tf.sparse.sparse_dense_matmul operation, which are subsequently applied
                independent of the actual batch size. Defaults to None, then an attempt is made to infer this from the
                input, which may cause an error.
        """
        super(HealpySmoothing, self).__init__()

        # pixels
        self.nside = nside
        self.indices = indices
        self.nest = nest
        self.mask = mask

        # smoothing
        assert fwhm is not None or sigma is not None, f"One of fwhm and sigma has to be specified"
        assert fwhm is None or sigma is None, f"Only one of fwhm and sigma can be specified"

        self.fwhm = fwhm
        self.sigma = sigma
        self.n_sigma_support = n_sigma_support
        self.arcmin = arcmin
        self.per_channel_repetitions = per_channel_repetitions
        self.data_path = data_path
        self.max_batch_size = max_batch_size

        if self.fwhm == 0.0 or self.sigma == 0.0:
            self.do_smoothing = False
            print(f"The layer implements the identity, smoothing is disabled")
        else:
            self.do_smoothing = True

            if isinstance(self.fwhm, (list, np.ndarray)):
                assert (
                    self.per_channel_repetitions is None
                ), f"per_channel_repetitions can't be specified when fwhm is a list, since it is then inferred"

                self.fwhm = np.array(self.fwhm)

                # smallest smoothing scale from which the others are derived by looping
                fwhm_min = np.min(self.fwhm)

                # ceil to be conservative, square because Gaussian variances are added (not stds)
                self.per_channel_repetitions = np.ceil((self.fwhm / fwhm_min) ** 2).astype(int)
                self.fwhm = fwhm_min

            elif isinstance(self.sigma, (list, np.ndarray)):
                assert (
                    self.per_channel_repetitions is None
                ), f"per_channel_repetitions can't be specified when sigma is a list, since it is then inferred"

                self.sigma = np.array(self.sigma)
                sigma_min = np.min(self.sigma)
                self.per_channel_repetitions = np.ceil((self.sigma / sigma_min) ** 2).astype(int)
                self.sigma = sigma_min

            elif isinstance(self.per_channel_repetitions, list):
                self.per_channel_repetitions = np.array(self.per_channel_repetitions)

            # internally, the smoothing is always done with sigma
            if self.sigma is None:
                self.sigma = self.fwhm / np.sqrt(8 * np.log(2))

            # angle conversions
            if self.arcmin:
                self.sigma_arcmin = self.sigma
                self.sigma_rad = self._arcmin_to_rad(self.sigma_arcmin)
            else:
                self.sigma_rad = self.sigma
                self.sigma_arcmin = self._rad_to_arcmin(self.sigma_rad)

            self.fwhm_arcmin = self.sigma_arcmin * np.sqrt(8 * np.log(2))

            # derived attributes
            self.n_indices = len(indices)
            self.kernel_func = lambda r: np.exp(-0.5 / self.sigma_rad**2 * r**2)
            self.file_label = f"-nside{self.nside}-sigma{self.sigma_arcmin:4.2f}-n_sigma{n_sigma_support}"

            if self.per_channel_repetitions is not None:
                per_channel_factor = np.sqrt(self.per_channel_repetitions)
                print(f"Using the per channel smoothing repetitions {self.per_channel_repetitions}")
                print(
                    f"Using the per channel smoothing scales "
                    f"sigma = {per_channel_factor * self.sigma_arcmin} arcmin, "
                    f"fwhm = {per_channel_factor * self.fwhm_arcmin} arcmin"
                )
            else:
                print(
                    f"Using the per channel smoothing scale sigma = {self.sigma_arcmin:4.2f} arcmin, "
                    f" fwhm = {self.fwhm_arcmin:4.2f} arcmin"
                )

            if self.data_path is not None:
                try:
                    self.ind_coo = np.load(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"))
                    self.val_coo = np.load(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"))
                    print(f"Successfully loaded sparse kernel indices and values from {self.data_path}")
                except FileNotFoundError:
                    self._build_tree()
                    self._build_kernel()
            else:
                self._build_tree()
                self._build_kernel()

            self._build_sparse_tensor()
            print(f"Successfully created the sparse kernel tensor")

    def build(self, input_shape: tuple) -> None:
        """Checks whether the input shape is compatible with the initialized layer. Note that the sparse-dense matrix
        multiplication might be split into multiple operations, depending on the nonzero entries in the sparse kernel
        matrix and batch dimension.

        Args:
            input_shape (tuple): Shape of the input, which is expected to be (n_batch, n_indices, n_channels).
        """
        if self.do_smoothing:
            # batch dimension
            if self.max_batch_size is not None:
                self.n_batch = self.max_batch_size
            elif input_shape[0] is not None:
                self.n_batch = input_shape[0]
            else:
                self.n_batch = None
                print(
                    f"Since the batch size cannot be inferred from the input shape and max_batch_size is not "
                    f"available, no sparse-dense matmul splits are performed, which may cause an error."
                )

            # map dimensions
            assert self.n_indices == input_shape[1]
            self.n_channels = input_shape[2]

            if self.per_channel_repetitions is not None:
                assert (
                    len(self.per_channel_repetitions) == self.n_channels
                ), f"The list per_channel_repetitions has to have length {self.n_channels}"

                assert (
                    self.per_channel_repetitions.dtype == int
                ), f"The list per_channel_repetitions has to contain integers only"

            if self.mask is not None:
                self.mask = tf.cast(self.mask, dtype=tf.float32)
                if tf.rank(self.mask).numpy() == 1:
                    self.mask = tf.expand_dims(self.mask, axis=0)
                    self.mask = tf.expand_dims(self.mask, axis=-1)
                elif tf.rank(self.mask).numpy() == 2:
                    self.mask = tf.expand_dims(self.mask, axis=0)

                assert (
                    self.mask.shape[1] == self.n_indices
                ), f"The mask has to have shape (1, n_indices, 1) or (1, n_indices, n_channels)"

            self.n_matmul_splits = 1
            # check if we need to split the matmul
            if self.n_batch is not None:
                while not (
                    # tf.split only does even splits for integer arguments
                    (self.n_batch % self.n_matmul_splits == 0)
                    and
                    # due to the int32 limitation of tf.sparse.sparse_dense_matmul
                    (self.n_matmul_splits >= self.n_batch * len(self.sparse_kernel.indices) / 2**31)
                ):
                    self.n_matmul_splits += 1

            print(f"Successfully built the smoothing layer")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer on the input tensor.

        Args:
            inputs (tf.Tensor): Tensor of shape (n_batch, n_indices, n_channels).

        Returns:
            tf.Tensor: Smoothed output tensor of identical shape.
        """
        if self.do_smoothing:
            # (n_indices, n_batch, n_channels)
            indices_first = tf.transpose(inputs, (1, 0, 2))

            # list of (n_indices, n_batch)
            separate_channels = tf.unstack(indices_first, axis=2)

            stack = []
            for i, single_channel in enumerate(separate_channels):
                if self.per_channel_repetitions is not None:
                    for _ in range(self.per_channel_repetitions[i]):
                        single_channel = utils.split_sparse_dense_matmul(
                            self.sparse_kernel, single_channel, self.n_matmul_splits
                        )
                else:
                    single_channel = utils.split_sparse_dense_matmul(
                        self.sparse_kernel, single_channel, self.n_matmul_splits
                    )

                stack.append(single_channel)

            # (n_indices, n_batch, n_channels)
            channels_last = tf.stack(stack, axis=2)

            # (n_batch, n_indices, n_channels)
            channels_last = tf.transpose(channels_last, (1, 0, 2))

            if self.mask is not None:
                channels_last *= self.mask

            return channels_last

        else:
            return inputs

    def _build_tree(self) -> None:
        """Builds a BallTree to find the nearest neighbors of each pixel. The number of neighbors is determined by the
        radius n_sigma_support * sigma. The maximum number of neighbors is determined by the pixel with the most
        neighbors within that radius. The Gaussian smoothing kernel is evaluated at the distances to the neighbors.
        """
        print(
            f"Creating tree for {self.n_indices} pixels and radius n_sigma_support * sigma = "
            f"{self.sigma_arcmin * self.n_sigma_support:4.2f} arcmin"
        )

        lon, lat = hp.pix2ang(self.nside, ipix=self.indices, nest=self.nest, lonlat=True)
        theta = np.stack([np.radians(lat), np.radians(lon)], axis=1)

        tree = BallTree(theta, metric="haversine")

        # determine the maximum number of neighbors
        inds_r = tree.query_radius(theta, r=self.sigma_rad * self.n_sigma_support)
        n_neighbours = [len(i) for i in inds_r]
        self.max_neighbors = np.max(n_neighbours)
        print(f"The maximal number of neighbors within that radius is {self.max_neighbors}")

        # find the per pixel k nearest neighbors
        n_theta_splits = 100
        theta_split = np.array_split(theta, n_theta_splits)
        list_dist_k, list_inds_k = [], []
        for theta_ in tqdm(theta_split, total=n_theta_splits, desc="querying the tree"):
            dist_k, inds_k = tree.query(theta_, k=self.max_neighbors, return_distance=True, sort_results=True)
            list_dist_k.append(dist_k)
            list_inds_k.append(inds_k)

        dist_k = np.concatenate(list_dist_k, axis=0)
        self.inds_k = np.concatenate(list_inds_k, axis=0, dtype=np.int64)
        self.kernel_k = self.kernel_func(dist_k).astype(np.float32)

    def _build_kernel(self) -> None:
        """Builds the indices and values of the coo sparse kernel matrix as dense arrays, which may be stored to disk.
        """
        # row, all of the pixels in the patch
        inds_r = tf.constant(np.arange(self.n_indices), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.repeat(inds_r, self.max_neighbors, axis=1)

        # column, all of the pixels that we want to sum over
        inds_c = tf.constant(self.inds_k, dtype=tf.int64)

        # shape (n_non_zero, 2)
        self.ind_coo = tf.concat([tf.reshape(inds_r, (-1, 1)), tf.reshape(inds_c, (-1, 1))], axis=1)

        # shape(n_non_zero,)
        self.val_coo = tf.reshape(self.kernel_k, (-1,))

        if self.data_path is not None:
            np_ind_coo = self.ind_coo.numpy()
            np_val_coo = self.val_coo.numpy()
            print(
                f"Storing sparse kernel indices ({np_ind_coo.nbytes/1e9:4.2f} GB, dtype {np_ind_coo.dtype}) and "
                f"values ({np_val_coo.nbytes/1e9:4.2f} GB, dtype {np_val_coo.dtype})"
            )

            os.makedirs(self.data_path, exist_ok=True)
            np.save(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"), np_ind_coo)
            np.save(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"), np_val_coo)

    def _build_sparse_tensor(self) -> None:
        """Builds the tf.sparse.SparseTensor from the dense indices and values."""
        self.sparse_kernel = tf.sparse.SparseTensor(
            indices=self.ind_coo,
            values=self.val_coo,
            dense_shape=(self.n_indices, self.n_indices),
        )
        self.sparse_kernel = tf.sparse.reorder(self.sparse_kernel)

        # the kernel entries within rows have to sum to one
        col_sum = tf.sparse.reduce_sum(self.sparse_kernel, axis=1, output_is_sparse=False)
        self.sparse_kernel = self.sparse_kernel / tf.expand_dims(col_sum, axis=0)

        del self.ind_coo
        del self.val_coo

    @staticmethod
    def _rad_to_arcmin(theta):
        return theta / np.pi * (180 * 60)

    @staticmethod
    def _arcmin_to_rad(theta):
        return theta * np.pi / (60 * 180)
