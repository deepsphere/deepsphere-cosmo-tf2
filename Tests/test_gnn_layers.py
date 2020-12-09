import pytest
import numpy as np
import tensorflow as tf
import healpy as hp

from deepsphere import gnn_layers


def test_Chebyshev():
    # this is the result from Deepsphere with tf 1.x
    result = np.array([[[-1.0755178 , -0.38834727, -1.8771361 ],
                        [-0.717345  , -0.05060194,  0.5165049 ],
                        [-0.26436844, -1.9551289 ,  1.5731683 ]],

                       [[ 0.20308694,  0.18807065,  0.5316967 ],
                        [-0.15023257,  0.5063435 , -2.1237612 ],
                        [ 0.8385707 , -0.05848193, -0.99261296]],

                       [[-0.72604334, -1.1806095 , -1.2536838 ],
                        [ 0.94199276, -0.45402163, -0.37584513],
                        [-0.5897862 , -0.99283355,  0.39791816]],

                       [[ 0.35315517, -0.69710857,  0.89600056],
                        [-1.2446954 ,  0.17451617,  2.5738995 ],
                        [-0.48410773, -1.3228154 ,  0.73148715]],

                       [[-0.47506812,  0.07938811, -0.7293904 ],
                        [ 0.9361354 ,  0.5696295 ,  1.4792094 ],
                        [-0.5101731 ,  0.43999135,  0.34349984]]], dtype=np.float32)

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

    assert np.all(np.abs(new.numpy() - result) < 1e-5)

    # same with new activation
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    new = cheb(x)

    assert np.all(np.abs(new.numpy() - result) < 1e-5)

    # now we test bias and batch norm
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear",
                                use_bias=True, use_bn=True)
    new = cheb(x)


def test_Monimials():
    # this is the result from Deepsphere with tf 1.x
    result = np.array([[[ 0.04206353,  0.46168754,  0.10546149],
                        [-0.5492798 , -0.32608002,  0.5628096 ],
                        [-0.11329696, -0.7900159 ,  0.92530084]],

                       [[ 0.06915615,  0.03369189,  0.0245935 ],
                        [-0.89208144, -0.11626951, -0.10967396],
                        [ 0.01909873, -0.16593638, -0.1462554 ]],

                       [[-0.29119226, -0.12377091, -0.0128078 ],
                        [ 0.36727118,  0.30154356, -0.02591037],
                        [-0.23363924, -0.14655769,  0.3258103 ]],

                       [[ 0.00471622, -0.03371258,  0.00214787],
                        [ 0.31400114, -0.57628125,  1.5108933 ],
                        [ 0.09324764, -0.75300777,  0.40933472]],

                       [[ 0.12954447,  0.06049673,  0.15058015],
                        [ 0.38768154, -0.24916826,  0.43720144],
                        [-0.1512235 ,  0.01706326,  0.14433491]]], dtype=np.float32)

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

    assert np.all(np.abs(new.numpy() - result) < 1e-5)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu")
    new_1 = mon(x)

    assert new_1.shape == new.shape

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu", use_bn=True, use_bias=True)
    new_1 = mon(x)


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
