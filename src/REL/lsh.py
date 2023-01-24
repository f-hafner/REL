
import time 
import numpy as np 
import logging 
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
import pdb 
from scipy import sparse

# TODO:
    # document the class?
    # add the academic references


def k_shingle(s, k):
    "convert string s into shingles of length k"
    shingle = []
    for i in range(len(s) - k + 1):
        shingle.append(s[i:(i+k)])
    return shingle


def cols_to_int_multidim(a):
    """
    Combine columns in all rows to an integer: [[1,20,3], [1,4,10]] becomes [1203,1410].

    Notes
    ------
    Advantage: uses vectorized numpy to create a unique signature.
    Disadvantage: Because one additional row increases the size of the integer at least by an order of magnitude, 
    this only works for cases where the bands are not too large (otherwise integer overflow problems).

    In practice I have found that optimal bands are typically not long enough to cause problems.
    """
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
    Calculate groups of indices of unique rows in a multidimensional array with the same signature
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
        "Build a vocabulary of the shingles in a document."
        vocab = list(set([shingle for sublist in self.shingles for shingle in sublist]))
        self.vocab = vocab

    def encode_binary(self): 
        """
        Create sparse binary vectors for each mention.

        Output: CSR sparse matrix.
        Rows indicate mentions, columns indicate whether the mention contains the shingle. 
        """
        logging.debug("making one-hot vectors")
        binarizer = MultiLabelBinarizer(sparse_output=True)
        self.vectors = binarizer.fit_transform(self.shingles)


class LSHRandomProjections(LSHBase):
    """
    A class for locality-sensitive hashing with random projections.
    

    Parameters:
    -----------
    mentions:
    shingle_size:
    signature_size:
    band_length:
    seed:     
    """
    # TODO: document more 

    def __init__(self, mentions, shingle_size, signature_size, band_length, seed=3):
        # sparse_binary: should the sparse 0/1 matrix be stored with scipy sparse? takes less memory.
        super().__init__(mentions, shingle_size)
        if signature_size % band_length != 0:
            raise ValueError("Signature needs to be divisible into equal-sized bands.")
        self.signature_size = signature_size 
        self.band_length = band_length 
        self.rng = np.random.default_rng(seed=seed)
    
    def make_signature(self):
        """
        Create a signature for a given mention, using random projections.
        """
        logging.debug(f"Making signature. vectors shape is {self.vectors.shape}")
        # TODO: can this be more memory-efficient by generating directly the scipy sparse function? 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html
        hyperplanes = self.rng.choice([-1, 1], (self.signature_size, self.vectors.shape[1]))
        hyperplanes = sparse.csr_matrix(hyperplanes)
        products = self.vectors.dot(hyperplanes.transpose()).toarray()
        # products = products.toarray()
        sign = 1 + (products > 0) # TODO: can I change the downstream function for this? now it should be much easier to transform the signatures into a single string?
        self.signature = sign

    def all_candidates_to_all(self):
        "Fall-back option to return the non-clustered input: each mention is a candidate coreference for all"
        n_mentions = self.vectors.shape[0]
        self.candidates = [set(range(n_mentions)) for _ in range(n_mentions)]

    def get_candidates(self):
        "Extract similar candidates for each mention by comparing subsets of the signature"
        logging.debug("getting candidates...")
        n_bands = int(self.signature_size / self.band_length)
        if self.vectors.shape[0] == 1:
            candidates = [set()]
            candidates[0].add(0)
        else:
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

    def cluster(self): 
        """
        Cluster mentions together based on their similarity. This is the main functionality of the LSH class.
        """
        start = time.time()
        logging.debug("building vocabulary")
        self._build_vocab()
        logging.debug("encoding to binary")
        self.encode_binary()
        logging.debug("making signature")

        if self.vectors.shape[1] == 0: # no signature possible b/c no mention is longer than the shingle size.
            logging.debug('self.vectors.shape[1] is 0.')
            self.all_candidates_to_all()
        else:
            self.make_signature()
            logging.debug("getting candidate groups")
            self.get_candidates()
        self.time = time.time() - start 

    def summarise(self):
        "Summarise the time taken and output from clustering one LSH instance."
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
