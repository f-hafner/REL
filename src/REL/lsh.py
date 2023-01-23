
from random import shuffle, seed 
import time 
import numpy as np 
import logging 
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
import pdb 
import sys 
from scipy import sparse
seed(3)


def k_shingle(s, k):
    "convert string s into shingles of length k"
    shingle = []
    for i in range(len(s) - k + 1):
        shingle.append(s[i:(i+k)])
    return shingle


# def cols_to_int(a):
#     "combine columns in all rows to an integer: [[1,20,3], [1,4,10]] becomes [1203,1410]"
#     existing_powers = np.floor(np.log10(a))
#     nrows, ncols = a.shape 

#     cumsum_powers = np.fliplr(np.cumsum(np.fliplr(existing_powers), axis=1))

#     add_powers = [x for x in reversed(range(ncols))]
#     add_powers = np.tile(add_powers, (nrows, 1))

#     mult_factor = cumsum_powers - existing_powers + add_powers  
#     summationvector = np.ones((ncols, 1)) 
#     out = np.matmul(a * 10**mult_factor, summationvector)
#     return out 



def idx_unique_multidim(a):
    "groups row indices in a multidimensional arrays by their unique signature"
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


# def reshape_rows_reps(a):
#     "reshape a 3-d array of n_reps x n_rows x n_cols to n_rows x n_reps x n_cols"
#     n_reps, n_rows, n_cols = a.shape
#     a = a.reshape(n_reps*n_rows, n_cols)
#     # extractor indices: for 3 reps, 2 rows: [0,2,4,1,3,5]. to reorder a
#         # in other words: goes from 0 to (n_reps * n_rows). step sizes are n_rows. starts are the row indices
#     idx = np.arange(n_reps*n_rows).reshape(n_reps, n_rows).T.reshape(-1,1)
#     a = np.take_along_axis(a, idx, axis=0)
#     a = a.reshape(n_rows, n_reps, n_cols)
#     return a 

# def minhash_signature_np(x, n_reps):
#     """Make a minhash signature of array x with length n_reps.

#     Inputs
#     ------
#     x: axis 0 are observations, columns are binary one-hot encoded vectors
#     """
#     # get indices 
#     indices = np.arange(x.shape[1])
#     rng = np.random.default_rng(12345) # TODO: this should be defined at class instantiation

#     # expand by n_reps 
#     indices_mult = np.tile(indices, (n_reps, 1)) # reorder the columns n_reps times 
#     x_mult = np.tile(x, (n_reps, 1)).reshape((n_reps,) + x.shape) # new shape: (n_resp, x.shape[0], x.shape[1

#     # permute indices and apply to x_mult
#     permuted_indices = rng.permuted(indices_mult, axis=1)
#     x_mult_permuted = np.take_along_axis(x_mult, permuted_indices[:, np.newaxis], 2)

#     # for the reduction below, need to have all samples of the same observation in one block
#     x_mult_permuted = reshape_rows_reps(x_mult_permuted)

#     # make signature
#     sig = x_mult_permuted.argmax(axis=2)
#     return sig 


# def signature_to_bucket(signature, n_bands):
#     "Collect items with same bands in buckets"
#     num_cols = signature.shape[0] # number of documents to classify
#     bands = np.split(signature, n_bands, axis=1)
#     buckets = []
#     for band in bands:
#         items_buckets = defaultdict(list)
#         items = np.vsplit(band, num_cols)
#         for i, item in enumerate(items): # this orders the row indices into groups that have the same signature 
#             item = tuple(item.flatten().astype(int)) 
#             items_buckets[item].append(i)  # assign row i to item--ie, groups observations into buckets with the same signature 
#         buckets.append(items_buckets)

#     return buckets

## new stuff
def cols_to_int_multidim(a):
    "combine columns in all rows to an integer: [[1,20,3], [1,4,10]] becomes [1203,1410]"
    existing_powers = np.floor(np.log10(a))
    n_bands, nrows, ncols = a.shape 

    cumsum_powers = np.fliplr(np.cumsum(np.fliplr(existing_powers), axis=1))

    add_powers = [x for x in reversed(range(ncols))]
    add_powers = np.tile(add_powers, (nrows, 1))

    mult_factor = cumsum_powers - existing_powers + add_powers  
    summationvector = np.ones((ncols, 1)) 
    out = np.matmul(a * 10**mult_factor, summationvector)
    return out 

def vectorize_signature_bands(a, n_bands, band_length):
    """ 
    Convert a signature array of dimension (n_items, signature_length) into an array of (n_bands, n_items, band_length).
    
    This is a vectorized version for np.vstack(np.split(a, indices_or_sections=n_bands, axis=1)). 
    The idea is to then use a vectorized function to extract the indices, instead of looping over each element in the output of np.split().
    """
    n_items, signature_length = a.shape
    
    # stacked bands of each item, stacked together
    stacked_bands = a.reshape(n_items*n_bands, band_length) 
    # reorder so that the first band of all items comes first, then the second band of all items, etc.
    reordering_vector = np.arange(n_items*n_bands).reshape(n_items, n_bands).T.reshape(1, -1)

    result = stacked_bands[reordering_vector, :].reshape(n_bands, n_items, band_length)
    
    return result 

# this replaces idx_multidim
def group_unique_indices(a):
    """
    calculate groups of indices of unique rows in a multidimensional array with the same signature
    the groups are returned by band.

    Returns a list of lists. One list corresponds to each band, and it indicates the rows
    of a that have the same band.
    """
    n_bands, n_items, length_band = a.shape
    a = cols_to_int_multidim(a).squeeze()
    
    sort_idx = np.argsort(a, axis=1) # necessary for later, need to calc anyway
    a_sorted = np.sort(a, axis=1) # faster alternative to np.take_along_axis(b, sort_idx, axis=1)

    # indicators for where a sequence of different unique elements starts 
    indicators = a_sorted[:, 1:] != a_sorted[:, :-1]
    first_element = np.tile([[True]], n_bands).T 
    unq_first = np.concatenate((first_element, indicators), axis=1)

    # calculate number of unique items 
    unq_count = [np.diff(np.nonzero(row)[0]) for row in unq_first] # iterate through rows.
    # split sorted array into groups of identical items. only keep groups with more than one item. 
    unq_idx = [[a for a in np.split(sort_idx[i], np.cumsum(count)) if len(a) > 1] for i, count in enumerate(unq_count)] 

    return unq_idx

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

    def encode_binary(self, sparse_output=True): # TODO: remove this argument 
        """Create binary vectors for each mention. 
        
        Parameters:
        ----------
        sparse_output: Argument passed to `sklearn.preprocessing.MultiLabelBinarizer()`.
        """
        logging.debug("making one-hot vectors")
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
        vectors = binarizer.fit_transform(self.shingles)
        self.vectors = vectors


class LSHMinHash(LSHBase):
    "LSH with MinHashing and numpy"

    def __init__(self, mentions, shingle_size, signature_size, band_length, sparse_binary=True):
        # sparse_binary: should the sparse 0/1 matrix be stored with scipy sparse? takes less memory.
        super().__init__(mentions, shingle_size)
        if signature_size % band_length != 0:
            raise ValueError("Signature needs to be divisible into equal-sized bands.")
        self.signature_size = signature_size 
        self.band_length = band_length 
        self.sparse_binary = sparse_binary
    
    def make_signature(self):
        "make array of dense vectors with MinHashing. each row is one mention"
        logging.debug(f"Making signature. vectors shape is {self.vectors.shape}")
        # pdb.set_trace()
        templist = []
        rng = np.random.default_rng(seed=3)
        i = 0
        if isinstance(self.vectors, np.ndarray):
            logging.debug("using binary numpy arrays")
            while i < self.signature_size:
                rng.shuffle(self.vectors, axis=1)
                sig_i = 1 + self.vectors.argmax(axis=1) # add one for the log10 operations in idx_unique_multidim 
                templist.append(sig_i)
                i += 1
            self.signature = np.stack(templist, axis=1)
        else: # older versions of scipy have not _coo attribute. TODO: fix this
        # elif isinstance(self.vectors, sparse._coo.coo_matrix):
            # not sure how efficient this is. switching a lot between data structures.
            logging.debug('using binary sparse matrices')
            rng = np.random.default_rng(seed=3) # TODO: put this to class instantiation
            # vectors = mylsh.vectors
            logging.debug("making hyperplanes")
            hyperplanes = rng.choice([-1, 1], (self.signature_size, self.vectors.shape[1]))
            # TODO: make vectors a csr matrix (?)
            hyperplanes = sparse.csr_matrix(hyperplanes)
            logging.debug("making dot product")
            products = self.vectors.dot(hyperplanes.transpose())
            logging.debug("making signature")
            products = products.toarray()
            sign = 1 + (products > 0) # TODO: can I change the downstream function for this? now it should be much easier to transform the signatures into a single string?
            self.signature = sign

            
    # def make_signature_np(self):
    #     signature = minhash_signature_np(self.vectors, self.signature_size)
    #     self.signature = signature + np.ones(signature.shape)  # this is for the log10 operations: do not want to have 0s

    def all_candidates_to_all(self):
        "fall-back option to return the non-clustered input: each mention is a candidate coreference for all"
        n_mentions = self.vectors.shape[0]
        self.candidates = [set(range(n_mentions)) for _ in range(n_mentions)]

    # def get_candidates(self): ## TODO: use itertools
    #     "extract similar candidates for each mention by comparing subsets of the signature"
    #     logging.debug("getting candidates...")
    #     n_bands = int(self.signature_size / self.band_length)
        
    #     if self.vectors.shape[0] == 1:
    #         candidates = [set()]
    #         candidates[0].add(0)
    #     else:
    #         bands = np.split(ary=self.signature, indices_or_sections=n_bands, axis=1)
    #         candidates = [set() for _ in range(self.vectors.shape[0])]
                        
    #         # if len(candidates) > 1:
    #         # TODO: can I speed this up? 
    #         for band in bands:
    #             groups = idx_unique_multidim(band)
    #             groups = [g for g in groups if g.shape[0] > 1]
    #             for g in groups:
    #                 g = list(g)
    #                 for i in g:
    #                     for j in g:
    #                         if i != j:
    #                             candidates[i].add(j)
    #         # else: # idx_unique_multidim above does not work when there is only one candidate
    #         #     candidates[0].add(0)

    #     self.candidates = candidates

    def get_candidates_new(self):
        "extract similar candidates for each mention by comparing subsets of the signature"
        logging.debug("getting candidates...")
        n_bands = int(self.signature_size / self.band_length)
        if self.vectors.shape[0] == 1:
            candidates = [set()]
            candidates[0].add(0)
        else:
            # bands = np.split(ary=self.signature, indices_or_sections=n_bands, axis=1)
            candidates = [set() for _ in range(self.vectors.shape[0])]

            bands = vectorize_signature_bands(self.signature, n_bands=n_bands, band_length=self.band_length)
            buckets_by_band = group_unique_indices(bands)
            groups = [tuple(i) for i in itertools.chain.from_iterable(buckets_by_band)] # flatten group; use tuple for applying set()
            groups = set(groups) # we only need the unique groups 

            for group in groups:
                for i in group:
                    candidates[i].update(group)

            [candidates[i].discard(i) for i in range(len(candidates))]
        self.candidates = candidates


    def cluster(self, numpy_signature=False, candidates="new"): # TODO: tidy this, only use the new function for getting candidates
        "find similar records for each mention"
        start = time.time()
        logging.debug("building vocabulary")
        self._build_vocab()
        logging.debug("encoding to binary")
        self.encode_binary(sparse_output=self.sparse_binary)
        logging.debug("making signature")
        if self.vectors.shape[1] == 0: # no signature possible b/c no mention is longer than the shingle size.
            print('self.vectors.shape[1] is 0.')
            self.all_candidates_to_all()
        else:
            if numpy_signature:
                self.make_signature_np()
            else:
                self.make_signature()
            logging.debug("getting candidate groups")
            if candidates == "old":
                self.get_candidates()
            elif candidates == "new": # this seems to be slower than the old approach 
                self.get_candidates_new()
        self.time = time.time() - start 

    def summarise(self):
        sizes = [len(g) for g in self.candidates]
        print(f"took {self.time} seconds for {len(self.candidates)} mentions")
        print(f"average, min, max cluster size: {round(sum(sizes)/len(sizes),2)}, {min(sizes)}, {max(sizes)}")

    def efficiency_gain_comparisons(self):
        """
        Compare number of comparisons made for coreference search with option "lsh" and option "all".
        Useful for understanding time complexity. 
        And to assess whether number of comparisons is meaningfully reduced
        """
        sizes = [len(g) for g in self.candidates]
        runtime_all = len(self.candidates)*len(self.candidates)
        runtime_lsh = len(self.candidates)*(sum(sizes)/len(sizes))
        print(f"LSH makes fraction {round(runtime_lsh/runtime_all, 2)} of comparisons relative to option all.")
