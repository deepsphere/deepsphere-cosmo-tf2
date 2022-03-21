import os
import tempfile

import pytest
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import healpy as hp

from deepsphere import healpy_layers as hp_nn
from deepsphere import HealpyGCNN


def test_HealpyGCNN_plotting():
    # create dir for plots
    os.makedirs("./tests/test_plots", exist_ok=True)

    # clear session
    tf.keras.backend.clear_session()

    # we get a random map
    nside_in = 256
    n_pix = hp.nside2npix(nside_in)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 1]).astype(np.float32)
    indices = np.arange(n_pix)

    # define some layers
    layers = [hp_nn.HealpyPseudoConv(p=1, Fout=4),
              hp_nn.HealpyPool(p=1),
              hp_nn.HealpyChebyshev(K=5, Fout=8),
              hp_nn.HealpyPseudoConv(p=2, Fout=16),
              hp_nn.HealpyMonomial(K=5, Fout=32),
              hp_nn.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5}),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(4)]

    tf.random.set_seed(11)
    model = HealpyGCNN(nside=nside_in, indices=indices, layers=layers)
    model.build(input_shape=(3, n_pix, 1))
    model.summary()

    with pytest.raises(ValueError):
        filters1 = model.get_gsp_filters(3)

    # get some filters
    filters1 = model.get_gsp_filters("chebyshev")
    filters2 = model.get_gsp_filters("gcnn__residual_layer")

    # plot some filters (coeff)
    ax = model.plot_chebyshev_coeffs("chebyshev")
    base_path, _ = os.path.split(__file__)
    plt.savefig(os.path.join(base_path, "test_plots/plot_chebyshev_coeffs_cheby5.png"))
    plt.clf()
    ax = model.plot_chebyshev_coeffs("gcnn__residual_layer")
    plt.savefig(os.path.join(base_path, "test_plots/plot_chebyshev_coeffs_res.png"))
    plt.clf()

    # plot some filters (spectral)
    ax = model.plot_filters_spectral("chebyshev")
    plt.savefig(os.path.join(base_path, "test_plots/plot_filters_spectral_cheby5.png"))
    plt.clf()
    ax = model.plot_filters_spectral("gcnn__residual_layer")
    plt.savefig(os.path.join(base_path, "test_plots/plot_filters_spectral_res.png"))
    plt.clf()

    # plot some filters (section)
    figs = model.plot_filters_section("chebyshev", ind_in=[0], ind_out=[0])
    figs[0].savefig(os.path.join(base_path, "test_plots/plot_filters_section_cheby5.png"))
    plt.clf()
    figs = model.plot_filters_section("gcnn__residual_layer", ind_in=[0], ind_out=[0])
    figs[0].savefig(os.path.join(base_path, "test_plots/plot_filters_section_res_1.png"))
    plt.clf()

    # plot some filters (gnomonic)
    figs = model.plot_filters_gnomonic("chebyshev", ind_in=[0], ind_out=[0])
    figs[0].savefig(os.path.join(base_path, "test_plots/plot_filters_gnomonic_cheby5.png"))
    plt.clf()
    figs = model.plot_filters_gnomonic("gcnn__residual_layer", ind_in=[0,1,2], ind_out=[0])
    figs[0].savefig(os.path.join(base_path, "test_plots/plot_filters_gnomonic_res_1.png"))
    plt.clf()

    # get the output
    out = model(m_in)

    assert out.numpy().shape == (3, 4)


def test_HealpyGCNN():
    # clear session
    tf.keras.backend.clear_session()

    # we get a random map
    nside_in = 256
    n_pix = hp.nside2npix(nside_in)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 1]).astype(np.float32)
    indices = np.arange(n_pix)


    # define some layers
    layers = [hp_nn.HealpyPseudoConv(p=1, Fout=4),
              hp_nn.HealpyPool(p=1),
              hp_nn.HealpyChebyshev(K=5, Fout=8),
              hp_nn.HealpyChebyshev(K=5, Fout=8),
              hp_nn.Healpy_ViT(p=2, key_dim=8, num_heads=2, n_layers=3),
              hp_nn.HealpyPseudoConv_Transpose(p=2, Fout=16),
              hp_nn.HealpyPseudoConv(p=2, Fout=16),
              hp_nn.HealpyMonomial(K=5, Fout=32),
              hp_nn.HealpyMonomial(K=5, Fout=32),
              hp_nn.Healpy_Transformer(key_dim=8, num_heads=4),
              hp_nn.Healpy_Transformer(key_dim=8, num_heads=4, n_layers=2),
              hp_nn.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5}),
              hp_nn.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5}),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(4)]

    tf.random.set_seed(11)
    model = HealpyGCNN(nside=nside_in, indices=indices, layers=layers)
    model.build(input_shape=(3, n_pix, 1))
    model.summary(line_length=128)

    out = model(m_in)

    assert out.numpy().shape == (3,4)

    # now we check if we can save this
    with tempfile.TemporaryDirectory() as tempdir:
        # save the current weight
        model.save_weights(tempdir)

        # create new model
        tf.random.set_seed(12)
        model = HealpyGCNN(nside=nside_in, indices=indices, layers=layers)
        model.build(input_shape=(3, n_pix, 1))
        out_new = model(m_in)

        # output should be different
        assert not np.all(np.isclose(out.numpy(), out_new.numpy()))

        # restore weights
        model.load_weights(tempdir)

        # now it should be the same
        out_new = model(m_in)
        assert np.all(np.isclose(out.numpy(), out_new.numpy(), atol=1e-6))

    # test the use 4 graphing
    with pytest.raises(NotImplementedError):
        model = HealpyGCNN(nside=nside_in, indices=indices, layers=layers, n_neighbors=12)

    # more channels
    tf.keras.backend.clear_session()

    # we get a random map
    nside_in = 256
    n_pix = hp.nside2npix(nside_in)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 2]).astype(np.float32)
    indices = np.arange(n_pix)

    # define some layers
    layers = [hp_nn.HealpyPseudoConv(p=1, Fout=4),
              hp_nn.HealpyPool(p=1),
              hp_nn.HealpyChebyshev(K=5, Fout=8),
              hp_nn.HealpyPseudoConv(p=2, Fout=16),
              hp_nn.HealpyPseudoConv_Transpose(p=2, Fout=16),
              hp_nn.HealpyPseudoConv(p=2, Fout=16),
              hp_nn.HealpyMonomial(K=5, Fout=32),
              hp_nn.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5}),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(4)]

    tf.random.set_seed(11)
    model = HealpyGCNN(nside=nside_in, indices=indices, layers=layers)
    model.build(input_shape=(3, n_pix, 2))
    model.summary(line_length=128)

    out = model(m_in)

    assert out.numpy().shape == (3, 4)
