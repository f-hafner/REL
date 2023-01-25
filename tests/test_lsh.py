#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import REL.lsh as lsh 
import numpy as np
import itertools 


def test_k_shingle():
    output = lsh.k_shingle("random string", 5)
    expected = ["rando", "andom", "ndom ", "dom s", "om st", "m str", " stri", "strin", "tring"]
    assert output == expected, "shingles not built correctly"


def test_cols_to_int_multidim():
    a = np.array([[[1, 20, 3], [1, 4, 10]],
             [[1, 3, 5], [100, 3, 50]]]
            )
    output = lsh.cols_to_int_multidim(a) 
    expected = np.array(
        [
            [[1203], [1410]],
            [[135], [100350]]
        ]
    )
    assert np.all(output == expected), "rows do not convert correctly to integer"

def test_signature_to_3d_bands():
    a = np.array(
        [
            [1, 4, 7, 8, 10, 8], 
            [5, 3, 2, 6, 11, 0], 
            [1, 4, 2, 6, 13, 15]
        ]
    )

    n_bands = 2
    n_items = a.shape[0]
    band_length = int(a.shape[1]/n_bands)
    result = lsh.signature_to_3d_bands(a, n_bands=n_bands, band_length=band_length)

    expected = np.vstack(np.split(a, n_bands, axis=1)).reshape(n_bands, n_items, -1)
    assert np.all(result == expected), "signature not correctly converted to 3d bands"

def test_group_unique_indices():
    a = np.array([[[1, 4], [1, 4], [5,3], [5, 3], [1 , 2]],
                    [[7,8], [2, 7], [2, 7], [7, 8], [10, 3]]
                  ]) 
    output = lsh.group_unique_indices(a)

    # build expected
    groups_band0 = [[0, 1], [2, 3]]
    groups_band1 = [[1, 2], [0, 3]] 
    # Notes:
    # [1,2], [10,3] are not listed because their group is of size 1. 
    # [2,7] is before [7, 8] because 27 < 78
    groups_band0 = [np.array(i) for i in groups_band0]
    groups_band1 = [np.array(i) for i in groups_band1]
    expected = [groups_band0, groups_band1]

    o = itertools.chain.from_iterable(output)
    e = itertools.chain.from_iterable(expected)

    # test 
    assert all([np.all(i==j) for i, j in zip(o, e)]), "unique indices not grouped correctly"

def test_cluster_short_mentions():
    mentions = ['EEC', 'ABC']
    max_length = max([len(m) for m in mentions])
    mylsh = lsh.LSHRandomProjections(
        mentions=mentions, 
        shingle_size=max_length + 1, 
        n_bands=15)
    mylsh.cluster()
    expected = [set((0, 1)), set((0, 1))]
    assert expected == mylsh.candidates, \
        "lsh fails when shingle size longer than longest input mentions"
