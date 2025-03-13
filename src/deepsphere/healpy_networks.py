import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from pygsp import filters
from pygsp.graphs import SphereHealpix
from tensorflow.keras.models import Sequential

from . import gnn_layers as gnn
from . import healpy_layers as hp_nn
from . import logger
from . import plot


class HealpyGCNN(Sequential):
    """
    A graph convolutional network using the Keras model API and the layers from the model
    """

    def __init__(self, nside, indices, layers, n_neighbors=8, max_batch_size=None, initial_Fin=None):
        """
        Initializes a graph convolutional neural network using the healpy pixelization scheme
        :param nside: integeger, the nside of the input
        :param indices: 1d array of inidices, corresponding to the pixel ids of the input of the NN
        :param layers: a list of layers that will make up the neural network
        :param n_neighbors: Number of neighbors considered when building the graph, currently supported values are:
                            8 (default), 20, 40 and 60.
        :param max_batch_size: Maximal batch size this network is supposed to handle. This determines the number of
                                splits in the tf.sparse.sparse_dense_matmul operation, which are subsequently applied
                                independent of the actual batch size. Defaults to None, then no such precautions are
                                taken, which may cause an error.
        :param initial_Fin: Initial number of input features. Defaults to None, then like for max_batch_size, there
                            are no precautions in the tf.sparse.sparse_dense_matmul operation taken.
        """
        # This is necessary for every Layer
        super(HealpyGCNN, self).__init__(name="")

        logger.info("WARNING: This network assumes that everything concerning healpy is in NEST ordering...")

        if n_neighbors not in [8, 20, 40, 60]:
            raise NotImplementedError(
                f"The requested number of neighbors {n_neighbors} is nor supported. Choose " f"either 8, 20, 40 or 60."
            )

        # save the variables
        self.nside_in = nside
        self.indices_in = indices
        self.layers_in = layers
        self.n_neighbors = n_neighbors

        # first we check the consistency, by getting the total reduction factor of the nside
        self.reduction_fac = 1.0
        for layer in self.layers_in:
            if isinstance(layer, (hp_nn.HealpyPool, hp_nn.HealpyPseudoConv, hp_nn.Healpy_ViT)):
                self.reduction_fac *= 2 ** (layer.p)
            if isinstance(layer, hp_nn.HealpyPseudoConv_Transpose):
                self.reduction_fac /= 2 ** (layer.p)

        self.nside_out = int(self.nside_in // self.reduction_fac)
        if self.nside_out < 1:
            raise ValueError(
                "With the given input, the layers would reduce the nside below zero!"
                "Use less layers that reduce the nside, e.g. HealpyPool or HealpyPseudoConv..."
            )
        if not hp.isnsideok(self.nside_out, nest=True):
            raise ValueError(f"The ouput of the network does not have a valid nside {self.nside_out}...")

        logger.info(
            f"Detected a reduction factor of {self.reduction_fac}, the input with nside {self.nside_in} will be "
            f"transformed to {self.nside_out} during a forward pass. Checking for consistency with indices...",
        )

        # now we check if this makes sense with the given indices set
        mask_in = np.zeros(hp.nside2npix(self.nside_in))
        mask_in[indices] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=self.nside_out, order_in="NEST", order_out="NEST")
        mask_out[mask_out > 1e-12] = 1.0
        mask_in = hp.ud_grade(map_in=mask_out, nside_out=self.nside_in, order_in="NEST", order_out="NEST")
        transformed_indices = np.arange(hp.nside2npix(self.nside_in))[mask_in > 1e-12]

        if not np.all(np.sort(transformed_indices.astype(int)) == np.sort(self.indices_in.astype(int))):
            raise ValueError(
                "With the given indices it would not be possible to properly reduce the input maps "
                "with the reduction factor determined by the layers. Use the function "
                "<extend_indices> from utils with the determined minimal nside to make your set of "
                "indices compatible..."
            )
        else:
            logger.info("indices seem consistent...")

        # now we build the actual layers
        self.layers_use = []
        current_nside = self.nside_in
        current_indices = indices

        # the feature dimension of the input is only known here if it is explicitly specified
        current_Fin = initial_Fin

        for layer in self.layers_in:
            if isinstance(
                layer,
                (
                    hp_nn.HealpyChebyshev,
                    hp_nn.HealpyMonomial,
                    hp_nn.Healpy_ResidualLayer,
                    hp_nn.Healpy_Transformer,
                    hp_nn.HealpyBernstein,
                ),
            ):
                # we need to calculate the current L and get the actual layer
                sphere = SphereHealpix(
                    subdivisions=current_nside,
                    indexes=current_indices,
                    nest=True,
                    k=self.n_neighbors,
                    lap_type="normalized",
                )
                current_L = sphere.L
                current_A = sphere.A
                if isinstance(layer, hp_nn.Healpy_Transformer):
                    actual_layer = layer._get_layer(current_A)
                elif isinstance(
                    layer,
                    (hp_nn.HealpyChebyshev, hp_nn.HealpyMonomial, hp_nn.HealpyBernstein, hp_nn.Healpy_ResidualLayer),
                ):
                    if (max_batch_size is not None) and (current_Fin is not None):
                        n_matmul_splits = 1
                        while not (
                            # tf.split only does even splits for integer arguments
                            (max_batch_size * current_Fin % n_matmul_splits == 0)
                            # due to tf.sparse.sparse_dense_matmul
                            and (n_matmul_splits >= max_batch_size * current_Fin * len(current_L.indices) / 2**31)
                        ):
                            n_matmul_splits += 1
                        actual_layer = layer._get_layer(current_L, n_matmul_splits)

                    else:
                        actual_layer = layer._get_layer(current_L)
                else:
                    actual_layer = layer._get_layer(current_L)
                self.layers_use.append(actual_layer)
            elif isinstance(layer, (hp_nn.HealpyPool, hp_nn.HealpyPseudoConv, hp_nn.Healpy_ViT)):
                # a reduction is happening
                new_nside = int(current_nside // 2**layer.p)
                current_indices = self._transform_indices(
                    nside_in=current_nside, nside_out=new_nside, indices=current_indices
                )
                current_nside = new_nside
                self.layers_use.append(layer)
            elif isinstance(layer, hp_nn.HealpyPseudoConv_Transpose):
                # a reduction is happening
                new_nside = int(current_nside * 2**layer.p)
                current_indices = self._transform_indices(
                    nside_in=current_nside, nside_out=new_nside, indices=current_indices
                )
                current_nside = new_nside
                self.layers_use.append(layer)
            else:
                self.layers_use.append(layer)

            try:
                current_Fin = layer.Fout
            except AttributeError:
                # don't update, this is for example the case for residual or pooling layers that have Fin = Fout
                pass

        # Now that we have everything we can super init...
        super(HealpyGCNN, self).__init__(layers=self.layers_use)

    def _transform_indices(self, nside_in, nside_out, indices):
        """
        Transforms a set of indices to an array of indices with a new nside. If the resulting map is smaller, it
        assumes that the reduction is sensible, i.e. all no new indices will be used during the reduction.
        :param nside_in: nside of the input indices
        :param nside_out: nside of the output indices
        :param indices: indices (pixel ids)
        :return: a new set of indices with nside_out
        """
        # simple case
        if nside_in == nside_out:
            return indices

        # down sample a binary mask
        mask_in = np.zeros(hp.nside2npix(nside_in))
        mask_in[indices] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=nside_out, order_in="NEST", order_out="NEST")
        transformed_indices = np.arange(hp.nside2npix(nside_out))[mask_out > 1e-12]

        return transformed_indices

    def _get_filter_coeffs(self, layer: gnn.Chebyshev, ind_in=None, ind_out=None):
        """
        Return the Chebyshev filter coefficients of a layer.
        :param layer: a Chebyshev5 layer
        :param ind_in: index(es) of the input filter(s) (default None, all the filters)
        :param ind_out: index(es) of the output filter(s) (default None, all the filters)
        :returns: the weights of the filter in the right shape
        """
        K, Fout = layer.K, layer.Fout
        trained_weights = layer.kernel.numpy()  # Fin*K x Fout
        # possible in res layers
        if Fout is None:
            # Fin == Fout = prod(
            Fout = int(np.sqrt(np.prod(trained_weights.shape) // K))
        trained_weights = trained_weights.reshape((-1, K, Fout))

        # Fin x K x Fout => K x Fout x Fin
        trained_weights = trained_weights.transpose([1, 2, 0])
        if ind_in:
            trained_weights = trained_weights[:, :, ind_in]
        if ind_out:
            trained_weights = trained_weights[:, ind_out, :]
        return trained_weights

    def get_gsp_filters(self, layer, ind_in=None, ind_out=None, return_weights=False):
        """
        Get the filter as a pygsp format
        :param layer: index (int) or name of the layer. Can be figured out with <logger.info_summary>.
        :param ind_in: index(es) of the input filter(s) (default None, all the filters)
        :param ind_out: index(es) of the output filter(s) (default None, all the filters)
        :param return_weights: just return a list of weights
        :return: a list of filter(s) or a list of weights if return_weights
        """

        # get the layer
        if isinstance(layer, int):
            tf_layer = self.get_layer(index=layer)
        elif isinstance(layer, str):
            tf_layer = self.get_layer(name=layer)
        else:
            raise ValueError("layer should be either string or int.")

        # check if the layer is actually the right type
        if isinstance(tf_layer, gnn.GCNN_ResidualLayer):
            if not (isinstance(tf_layer.layer1, gnn.Chebyshev) and isinstance(tf_layer.layer2, gnn.Chebyshev)):
                raise ValueError(
                    f"The requested layer ({layer}) is of type {type(tf_layer)}, but only "
                    f"Chebyshev5 or GCNN_ResidualLayer layers (with Chebyshev5 sublayers) "
                    f"are supported..."
                )
        elif not isinstance(tf_layer, gnn.Chebyshev):
            raise ValueError(
                f"The requested layer ({layer}) is of type {type(tf_layer)}, but only "
                f"Chebyshev5 or GCNN_ResidualLayer layers (with Chebyshev5 sublayers) "
                f"are supported..."
            )

        # we get the weights
        if isinstance(tf_layer, gnn.GCNN_ResidualLayer):
            # get the weights
            # logger.info(tf_layer.layer1.kernel)
            weight1 = self._get_filter_coeffs(tf_layer.layer1, ind_in=ind_in, ind_out=ind_out)
            weight2 = self._get_filter_coeffs(tf_layer.layer2, ind_in=ind_in, ind_out=ind_out)
            weights = [weight1, weight2]

            # get the size of the features
            n_features = tf_layer.layer1.L.shape[0]

        else:
            # get the weights and reshape
            weight1 = self._get_filter_coeffs(tf_layer, ind_in=ind_in, ind_out=ind_out)
            weights = [weight1]
            # get the size of the features
            n_features = tf_layer.L.shape[0]

        if return_weights:
            return weights

        # get the nside by comparing sizes
        nside = len(self.indices_in) // n_features
        reduction_fac = 0
        while nside != 1:
            nside = nside // 4
            reduction_fac += 1
        nside = int(self.nside_in // 2 ** (reduction_fac))

        gsp_filters = []
        for weight in weights:
            pygsp_graph = SphereHealpix(
                subdivisions=nside,
                indexes=np.arange(hp.nside2npix(nside)),
                nest=True,
                k=self.n_neighbors,
                lap_type="normalized",
            )
            # pygsp_graph = utils.healpix_graph(nside=nside)
            pygsp_graph.estimate_lmax()
            gsp_filters.append(filters.Chebyshev(pygsp_graph, weight))

        return gsp_filters

    def plot_chebyshev_coeffs(
        self, layer, ind_in=None, ind_out=None, ax=None, title="Chebyshev coefficients - layer {}"
    ):
        """
        Plot the Chebyshev coefficients of a layer.
        layer : index (int) or name of the layer. Can be figured out with <logger.info_summary>.
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        title : figure title
        """
        weights = self.get_gsp_filters(layer, ind_in, ind_out, return_weights=True)
        if ax is None:
            ax = plt.gca()

        for weight in weights:
            K, Fout, Fin = weight.shape
            ax.plot(weight.reshape((K, Fin * Fout)), ".")
            ax.set_title(title.format(layer))
        return ax

    def plot_filters_spectral(self, layer, ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter of a special layer in the spectral domain.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """

        filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)
        if ax is None:
            ax = plt.gca()
        for filter in filters:
            filter.plot(sum=False, ax=ax, **kwargs)

        return ax

    def plot_filters_section(self, layer, ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter section on the sphere

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """

        filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)

        # get the layer (for K)
        if isinstance(layer, int):
            tf_layer = self.get_layer(index=layer)
        elif isinstance(layer, str):
            tf_layer = self.get_layer(name=layer)
        if isinstance(tf_layer, gnn.Chebyshev):
            K = tf_layer.K
        else:
            K = tf_layer.layer1.K

        figs = []
        for filter in filters:
            figs.append(plot.plot_filters_section(filter, order=K, **kwargs))
        return figs

    def plot_filters_gnomonic(self, layer, ind_in=None, ind_out=None, **kwargs):
        """Plot the filter localization on gnomonic view.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """

        filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)

        # get the layer (for K)
        if isinstance(layer, int):
            tf_layer = self.get_layer(index=layer)
        elif isinstance(layer, str):
            tf_layer = self.get_layer(name=layer)
        if isinstance(tf_layer, gnn.Chebyshev):
            K = tf_layer.K
        else:
            K = tf_layer.layer1.K

        figs = []
        for filter in filters:
            figs.append(plot.plot_filters_gnomonic(filter, order=K, **kwargs))

        return figs
