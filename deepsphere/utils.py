"""Utilities module."""

import numpy as np
from scipy import sparse
import healpy as hp


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
    M, M = L.shape
    I = sparse.identity(M, format='csr', dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= I
    return L
