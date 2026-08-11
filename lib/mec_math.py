import numpy as np
from bitarray import bitarray

def softmax(x, axis=-1):
    # Subtract the maximum value for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def normalize(v): # to probability distribution
    assert len(np.where(v < 0)[0]) == 0 # nonnegative
    s = v.sum()
    return v/s if s > 0 else v # else v=0

def entropy(p, axis=None, base=2):
    p = np.asarray(p, dtype=float)
    # 0*log(0) = 0
    mask = p > 0
    log_p = np.zeros_like(p)
    log_p[mask] = np.log(p[mask]) / np.log(base)
    return -np.sum(p * log_p, axis=axis)


def block_partition(data, block_size):
    bitdata = bitarray(data)
    # pad
    r = len(bitdata) % block_size
    bitdata += bitarray('0' * (block_size - r))
    assert len(bitdata) % block_size == 0

    blocks = []
    for i in range(0, len(bitdata), block_size):
        block = bitdata[i:i + block_size]
        blocks.append(block)
    return blocks


def barr_to_int(barr):
    val = 0
    for bit in barr:
        val = (val << 1) | bit
    return val


def int_to_barr(n: int, width: int = None) -> bitarray:
    if n == 0:
        bits = bitarray('0')
    else:
        bits = bitarray()
        while n > 0:
            bits.append(n & 1)  # extract least significant bit
            n >>= 1
        bits.reverse()

    # Pad with leading zeros if width is specified
    if width is not None:
        if len(bits) > width:
            raise ValueError(f"Integer too large to fit in {width} bits")
        pad = bitarray('0' * (width - len(bits)))
        bits = pad + bits

    return bits


def mec(p: np.array, q: np.array):
    """
    Algorithm 1:  https://arxiv.org/pdf/1611.04035.pdf
    Supports different sizes via padding to maximum dimension then trimming.
    """
    p = p.copy()
    q = q.copy()
    assert np.all(p >= 0) and np.all(q >= 0)
    if p.sum() != 1: p = normalize(p)
    if q.sum() != 1: q = normalize(q)

    d1, d2 = p.shape[0], q.shape[0]  # save original dimensions

    # equalize dimensions - extend smaller to larger
    if p.shape[0] > q.shape[0]:
        q = np.concatenate([q, np.zeros(p.shape[0] - q.shape[0]
                                        )])
    elif q.shape[0] > p.shape[0]:
        p = np.concatenate([p, np.zeros(q.shape[0] - p.shape[0]
                                        )])
    assert len(p) == len(q)

    # Joint distribution
    J = np.zeros((q.shape[0], p.shape[0]))
    M = np.stack((p, q), 0)
    r = M.max(axis=1).min()
    while r > 0:
        a_i = M.argmax(axis=1)
        M[0, a_i[0]] -= r
        M[1, a_i[1]] -= r
        J[a_i[0], a_i[1]] = r
        r = M.max(axis=1).min()
    return J[:d1, :d2]  # eliminate padding