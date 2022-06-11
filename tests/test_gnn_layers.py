import pytest
import numpy as np
import tensorflow as tf
import healpy as hp

from deepsphere import gnn_layers


def test_Chebyshev():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 1 / np.sqrt(7 * (K + 0.5) / 2)
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer)
    new = cheb(x)

    # same with new activation
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    new = cheb(x)

    # now we test bias and batch norm
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear",
                                use_bias=True, use_bn=True)
    new = cheb(x)


def test_Monimials():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 0.1
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation=tf.keras.activations.linear)
    new = mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu")
    new_1 = mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu", use_bn=True, use_bias=True)
    new_1 = mon(x)


def test_Bernstein():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 1 / np.sqrt(7 * (K + 0.5) / 2)
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer)
    new = bern(x)

    # same with new activation
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    new = bern(x)

    # now we test bias and batch norm
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear",
                                use_bias=True, use_bn=True)
    new = bern(x)


def test_GCNN_ResidualLayer():
    # we get a random map to pool
    n_pix = hp.nside2npix(4)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])

    # check exception
    with pytest.raises(IOError):
        _ = gnn_layers.GCNN_ResidualLayer("juhu", dict())

    # layer definition
    layer_type = "CHEBY"
    layer_kwargs = {"L": np.eye(n_pix, dtype=np.float64),
                    "K": 5,
                    "activation": tf.keras.activations.relu,
                    "regularizer": tf.keras.regularizers.l1}

    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu)
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    # same with batch norms
    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu, use_bn=True)
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu, use_bn=True,
                                              norm_type="layer_norm", bn_kwargs={"axis": (1,2)})
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    with pytest.raises(ValueError):
        res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                                  activation=tf.keras.activations.relu, use_bn=True,
                                                  norm_type="moving_norm")
