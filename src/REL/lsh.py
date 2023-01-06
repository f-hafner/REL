
from random import shuffle, seed 
import time 
import numpy as np 
seed(3)

def split_mention(m):
    return m.split(" ")

def k_shingle(s, k):
    "convert string s into shingles of length k"
    shingle = []
    for i in range(len(s) - k + 1):
        shingle.append(s[i:(i+k)])
    return shingle


def partition_signature(s, b):
    "Convert signature s into b partitions of equal size"
    assert len(s) % b == 0
    rg = int(len(s) / b)
    partitions = []
    for i in range(0, len(s), rg):
        v = s[i:i+rg]
        partitions.append(v)
    return partitions

def cols_to_int(a):
    "combine columns in all rows to an integer: [[1,20,3], [1,4,10]] becomes [1203,1410]"
    existing_powers = np.floor(np.log10(a))
    nrows, ncols = a.shape 

    cumsum_powers = np.fliplr(np.cumsum(np.fliplr(existing_powers), axis=1))

    add_powers = [x for x in reversed(range(ncols))]
    add_powers = np.tile(add_powers, (nrows, 1))

    mult_factor = cumsum_powers - existing_powers + add_powers  
    summationvector = np.ones((ncols, 1)) 
    out = np.matmul(a * 10**mult_factor, summationvector)
    return out 



def idx_unique_multidim(a):
    "groups rows in a multidimensional arrays by their unique signature"
    # a = cols_to_int(a).squeeze() # wrong
    # a = cols_to_string(a).squeeze() # slow 
    a = cols_to_int(a).squeeze()
    sort_idx = np.argsort(a)
    sort_idx
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1])) # "is the current value different from the previous?". the concat of [True]: because the first occurrence is always True (ie the first time it occur)
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0]) # np.nonzero(unq_first)[0] gives the indices of first elements in a_sorted
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return unq_idx


def reshape_rows_reps(a):
    "reshape a 3-d array of n_reps x n_rows x n_cols to n_rows x n_reps x n_cols"
    n_reps, n_rows, n_cols = a.shape
    a = a.reshape(n_reps*n_rows, n_cols)
    # extractor indices: for 3 reps, 2 rows: [0,2,4,1,3,5]. to reorder a
        # in other words: goes from 0 to (n_reps * n_rows). step sizes are n_rows. starts are the row indices
    idx = np.arange(n_reps*n_rows).reshape(n_reps, n_rows).T.reshape(-1,1)
    a = np.take_along_axis(a, idx, axis=0)
    a = a.reshape(n_rows, n_reps, n_cols)
    return a 

def minhash_signature_np(x, n_reps):
    """Make a minhash signature of array x with length n_reps.

    Inputs
    ------
    x: axis 0 are observations, columns are binary one-hot encoded vectors
    """
    # get indices 
    indices = np.arange(x.shape[1])
    rng = np.random.default_rng(12345)

    # expand by n_reps 
    indices_mult = np.tile(indices, (n_reps, 1)) # reorder the columns n_reps times 
    x_mult = np.tile(x, (n_reps, 1)).reshape((n_reps,) + x.shape) # new shape: (n_resp, x.shape[0], x.shape[1

    # permute indices and apply to x_mult
    permuted_indices = rng.permuted(indices_mult, axis=1)
    x_mult_permuted = np.take_along_axis(x_mult, permuted_indices[:, np.newaxis], 2)

    # for the reduction below, need to have all samples of the same observation in one block
    x_mult_permuted = reshape_rows_reps(x_mult_permuted)

    # make signature
    sig = x_mult_permuted.argmax(axis=2)
    return sig 


class LSHBase:
    # Important: order of occurences in shingles and vectors = order of input list (=order of occurrence in document)
    def __init__(self, mentions, shingle_size):
        if isinstance(mentions, dict):
            self.shingles = [k_shingle(m, shingle_size) for m in mentions.values()]
        elif isinstance(mentions, list):
            self.shingles = [k_shingle(m, shingle_size) for m in mentions]

    def _build_vocab(self):
        # shingles = [v["shingles"] for v in self.mentions.values()]
        vocab = list(set([shingle for sublist in self.shingles for shingle in sublist]))
        self.vocab = vocab

    def encode_binary(self, to_numpy=False):
        vectors = [[1 if word in cur_shingles else 0 for word in self.vocab] for cur_shingles in self.shingles]
        if not to_numpy:
            self.vectors = vectors 
        else:
            self.vectors = np.stack(vectors)


class LSHMinHash(LSHBase):
    "LSH with MinHashing and numpy"

    def __init__(self, mentions, shingle_size, signature_size, band_length):
        super().__init__(mentions, shingle_size)
        if signature_size % band_length != 0:
            raise ValueError("Signature needs to be divisible into equal-sized bands.")
        self.signature_size = signature_size 
        self.band_length = band_length 
    
    def make_signature(self):
        "make array of dense vectors with MinHashing. each row is one mention"
        templist = []
        rng = np.random.default_rng(seed=3)
        i = 0
        while i < self.signature_size:
            rng.shuffle(self.vectors, axis=1)
            sig_i = 1 + self.vectors.argmax(axis=1) # add one for the log10 operations in idx_unique_multidim 
            templist.append(sig_i)
            i += 1

        self.signature = np.stack(templist, axis=1)

    def make_signature_np(self):
        signature = minhash_signature_np(self.vectors, self.signature_size)
        self.signature = signature + np.ones(signature.shape)  # this is for the log10 operations: do not want to have 0s

    def all_candidates_to_all(self):
        "fall-back option to return the non-clustered input: each mention is a candidate coreference for all"
        n_mentions = self.vectors.shape[0]
        self.candidates = [set(range(n_mentions)) for _ in range(n_mentions)]

    def get_candidates(self):
        "extract similar candidates for each mention by comparing subsets of the signature"
        n_bands = int(self.signature_size / self.band_length)
        bands = np.split(ary=self.signature, indices_or_sections=n_bands, axis=1)
        candidates = [set() for _ in range(self.vectors.shape[0])]

        if len(candidates) > 1:
            for band in bands:
                groups = idx_unique_multidim(band)
                groups = [g for g in groups if g.shape[0] > 1]
                for g in groups:
                    g = list(g)
                    for i in g:
                        for j in g:
                            if i != j:
                                candidates[i].add(j)
        else: # idx_unique_multidim above does not work when there is only one candidate
            candidates[0].add(0)

        self.candidates = candidates

    def cluster(self, numpy_signature=False):
        "find similar records for each mention"
        start = time.time()
        self._build_vocab()
        self.encode_binary(to_numpy=True)
        if self.vectors.shape[1] == 0: # no signature possible b/c no mention is longer than the shingle size.
            self.all_candidates_to_all()
        else:
            if numpy_signature:
                self.make_signature_np()
            else:
                self.make_signature()
            self.get_candidates()
        self.time = time.time() - start 

    def summarise(self):
        sizes = [len(g) for g in self.candidates]
        print(f"took {self.time} seconds for {len(self.candidates)} mentions")
        print(f"average, min, max cluster size: {round(sum(sizes)/len(sizes),2)}, {min(sizes)}, {max(sizes)}")
