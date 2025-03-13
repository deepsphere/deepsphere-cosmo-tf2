"""Utilities module."""

import healpy as hp
import numpy as np
import tensorflow as tf
from scipy import sparse


def extend_indices(indices, nside_in, nside_out, nest=True):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always four pixels
    reduce naturally to a higher order pixel. Note that this function supports the ring ordering, however, since almost
    no other function does so, nest ordering is strongly recommended.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    # figire out the ordering
    if nest:
        ordering = "NEST"
    else:
        ordering = "RING"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, _ = L.shape
    identity = sparse.identity(M, format="csr", dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= identity
    return L


@tf.function
def split_sparse_dense_matmul(sparse_tensor, dense_tensor, n_splits=1):
    """
    Splits axis 1 of the dense_tensor such that tensorflow can handle the size of the computation.
    :param sparse_tensor: Input sparse tensor of rank 2.
    :param dense_tensor: Input dense tensor of rank 2.
    :param n_splits: Integer number of splits applied to axis 1 of dense_tensor.

    For reference, the error message to be avoided is:

    'Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]

    Call arguments received by layer "chebyshev" (type Chebyshev):
    • input_tensor=tf.Tensor(shape=(208, 7264, 128), dtype=float32)
    • training=False'
    """
    if n_splits > 1:
        print(
            f"Tracing... Due to tensor size, tf.sparse.sparse_dense_matmul is executed over {n_splits} splits."
            f" Beware of the resulting performance penalty."
        )
        dense_splits = tf.split(dense_tensor, n_splits, axis=1)
        result = []
        for dense_split in dense_splits:
            result.append(tf.sparse.sparse_dense_matmul(sparse_tensor, dense_split))
        result = tf.concat(result, axis=1)
    else:
        result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor)

    return result
