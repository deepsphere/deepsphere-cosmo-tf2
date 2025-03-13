import healpy as hp
import numpy as np
import tensorflow as tf
from pygsp.graphs import SphereHealpix

from deepsphere import gnn_transformers


def test_Graph_ViT():
    # create the input
    nside = 32
    n_pix = hp.nside2npix(nside)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])

    # create the layer
    tf.random.set_seed(11)
    p = 2
    key_dim = 16
    num_heads = 4
    graph_ViT = gnn_transformers.Graph_ViT(p=p, key_dim=key_dim, num_heads=num_heads, n_layers=3)
    output = graph_ViT(m_in)

    assert output.numpy().shape == (3, n_pix // 4**p, num_heads * key_dim)

    # same but wraped in TF function
    @tf.function()
    def call_layer(i):
        o = graph_ViT(i)
        return o

    output = call_layer(m_in)

    assert output.numpy().shape == (3, n_pix // 4**p, num_heads * key_dim)


def test_Graph_Transformer():
    # create the input
    nside = 8
    n_pix = hp.nside2npix(nside)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])
    A = SphereHealpix(subdivisions=8, nest=True, k=20, lap_type="normalized").A

    # create the layer
    tf.random.set_seed(11)
    key_dim = 16
    num_heads = 4
    graph_ViT = gnn_transformers.Graph_Transformer(A=A, key_dim=key_dim, num_heads=num_heads, n_layers=3)
    output = graph_ViT(m_in)

    assert output.numpy().shape == (3, n_pix, num_heads * key_dim)

    # same but wraped in TF function
    @tf.function()
    def call_layer(i):
        o = graph_ViT(i)
        return o

    output = call_layer(m_in)

    assert output.numpy().shape == (3, n_pix, num_heads * key_dim)
