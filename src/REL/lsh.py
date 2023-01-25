"""Implement a simple version of locality-sensitive hashing.

To deal with high-dimensional data (=many mentions), the class stores the feature vectors
as sparse matrices and uses random projections as hash functions. 

See chapter 3 in "Mining of Massive Datasets" (http://www.mmds.org/).
The time complexity is explained at the end of this video: https://www.youtube.com/watch?v=Arni-zkqMBA
(number of hyperplanes = band length).
The use of multiple bands is called amplification, which is discussed in the book 
but not in the video.
"""

import itertools
import logging 
import math 
import numpy as np 
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
import time 

# First, define a bunch of functions. TODO: should they be defined elsewhere? put in utils?

def k_shingle(s, k):
    "Convert string s into shingles of length k."
    shingle = []
    for i in range(len(s) - k + 1):
        shingle.append(s[i:(i+k)])
    return shingle


def cols_to_int_multidim(a):
    """Combine columns in all rows to an integer.
    
    For instance, [[1,20,3], [1,4,10]] becomes [1203,1410].

    Notes
    -----
    The advantage is that it uses vectorized numpy to collapse an
    entire row into one integer. The disadvantage is that one additional row increases 
    the size of the integer at least by an order of magnitude, which only works for cases where 
    the bands are not too large. But in practice, optimal bands are typically not long enough 
    to cause problems.

    :param a: 2-dimensional array
    :type a: np.ndarray
    :returns: An array of shape (n, 1), where the horizontally neighboring column values 
    are appended together.
    :rtype: np.ndarray
    """
    existing_powers = np.floor(np.log10(a))
    n_bands, nrows, ncols = a.shape 

    # sum existing powers from right to left
    cumsum_powers = np.flip(np.cumsum(np.flip(existing_powers, axis=2), axis=2), axis=2)
     
    add_powers = [x for x in reversed(range(ncols))]
    add_powers = np.tile(add_powers, (nrows, 1))

    mult_factor = cumsum_powers - existing_powers + add_powers  
    summationvector = np.ones((ncols, 1)) 
    out = np.matmul(a * 10**mult_factor, summationvector)
    return out 

def signature_to_3d_bands(a, n_bands, band_length):
    """Convert a signature from 2d to 3d.

    Convert a signature array of dimension (n_items, signature_length) into an array 
    of (n_bands, n_items, band_length).

    Notes
    -----
    This produces the same output as np.vstack(np.split(a, indices_or_sections=n_bands, axis=1)).
    When further processing the output, this is a useful alternative to looping on the output of
    np.split(a, indices_or_sections=n_bands, axis=1) because a single vectorized call can be used,
    while np.vstack(np.split(...)) is likely to be less efficient. 

    :param a: Array with 2 dimensions
    :type a: np.ndarray 
    :param n_bands: Number of bands the columns to cut into
    :type n_bands: int 
    :param band_length: Length of each band 
    :type band_length: int
    :returns: Array of shape (n_bands, n_items, band_length)
    :rtype: np.ndarray
    """
    n_items, signature_length = a.shape
    
    # stacked bands of each item, stacked together
    stacked_bands = a.reshape(n_items*n_bands, band_length) 
    # reorder so that the first band of all items comes first, then the second band of all items, etc.
    reordering_vector = np.arange(n_items*n_bands).reshape(n_items, n_bands).T.reshape(1, -1)
    
    result = stacked_bands[reordering_vector, :].reshape(n_bands, n_items, band_length)
    return result 

def group_unique_indices(a):
    """Compute indices of matching rows.

    In a 3-dimensional array, for each array (axis 0), 
    compute the indices of rows (axis=1) that are identical.
    Based on 1d-version here: https://stackoverflow.com/questions/23268605/grouping-indices-of-unique-elements-in-numpy

    :param a: 3-dimensional array
    :type a: np.ndarray
    :returns: List of lists. Outer lists correspond to bands. 
        Inner lists correspond to the row indices that 
        have the same values in their columns. An item 
        in the inner list is an np.array.
    :rtype: list
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

# ## Here follow the classes

class LSHBase:
    """
    Base class for locality-sensitive hashing.

    Attributes
    ----------
    shingle_size
        Size of shingles to be created from mentions
    mentions
        Mentions in which to search for similar items

    Methods
    -------
    encode_binary()
        One-hot encode mentions, based on shingles
    """
    # Important: order of occurences in shingles and vectors = order of input list (=order of occurrence in document)
    def __init__(self, mentions, shingle_size):
        """

        Parameters
        ----------
        :param mentions: Mentions in which to search for similar items
        :type mentions: list or dict
        :param shingle_size: Length of substrings to be created from mentions ("shingles")
        :type shingle_size: int
        """
        self.shingle_size = shingle_size
        if isinstance(mentions, dict):
            self.shingles = [k_shingle(m, shingle_size) for m in mentions.values()]
        elif isinstance(mentions, list):
            self.shingles = [k_shingle(m, shingle_size) for m in mentions]
        self._rep_items_not_show = ["shingles"] # do not show in __repr__ b/c too long

    def __repr__(self):
        items_dict_show = {k: v for k, v in self.__dict__.items() 
                                if k not in self._rep_items_not_show
                                and k[0] != "_" # omit private attributes
                            }
        items_dict_show = [f"{k}={v}" for k, v in items_dict_show.items()]
        return f"<{type(self).__name__}() with {', '.join(items_dict_show)}>"

    def _build_vocab(self):
        "Make vocabulary of unique shingles in all mentions."
        logging.debug("making vocabulary from shingles")
        vocab = list(set([shingle for sublist in self.shingles for shingle in sublist]))
        self.vocab = vocab

    def encode_binary(self): 
        """Create sparse binary vectors for each mention.

        :return: Indicator matrix. 
            Rows indicate mentions, columns indicate whether 
            the mention contains the shingle. 
        :rtype: scipy.sparse.csr_matrix
        """
        logging.debug("making one-hot vectors")
        binarizer = MultiLabelBinarizer(sparse_output=True)
        self.vectors = binarizer.fit_transform(self.shingles)


class LSHRandomProjections(LSHBase):
    """Class for locality-sensitive hashing with random projections.
    
    Attributes
    -----------
    mentions
        List or dict of mentions
    shingle_size
        Length of the shingles to be constructed from each string in `mentions`
    n_bands, band_length
        The signature of a mention will be n_bands*band_length.
        Longer bands increase precision, more bands increase recall. 
        If band_length is `None`, it is set as log(len(mentions)), which 
        will guarantee O(log(N)) time complexity.
    seed
        Random seed for np.random.default_rng

    Methods
    --------
    make_signature()
        Create a dense signature vector with random projections.
    get_candidates()
        Find groups of mentions overlapping signatures.
    cluster()
        End-to-end hashing from shingles to clusters.
        This is the main functionality of the class.
    summarise()
        Summarise time and output of cluster()
    efficiency_gain_comparisons()
        Compare number of computations for coreference search with hashing 
        and without hashing.
    """

    def __init__(self, mentions, shingle_size, n_bands, band_length=None, seed=3):
        """

        Parameters
        ----------
        :param mentions: Mentions in which to search for similar items
        :type mentions: list or dict
        :param shingle_size: Length of substrings to be created from mentions ("shingles")
        :type shingle_size: int
        :param n_bands: Number of signature bands (equal-sized cuts of the full signature)
        :type n_bands: int
        :param band_length: Length of bands
        :type band_length: int or None 
        :seed: Random seed for random number generator from numpy
        :type seed: int
        """
        super().__init__(mentions, shingle_size)
        self.seed = seed
        self.n_bands = n_bands
        if band_length is None:
            log_n_mentions = math.ceil(math.log(len(mentions))) # for O(log(N)) complexity
            self.band_length = max(1, log_n_mentions) # use 1 if exp(log(n_mentions)) < 1
        else:
            self.band_length = band_length
        self.signature_size = n_bands * self.band_length 
        self.rng = np.random.default_rng(seed=self.seed)
        self._rep_items_not_show.extend(["signature_size", "rng"])
    
    def make_signature(self):
        "Create a matrix of signatures with random projections."
        logging.debug(f"Making signature. vectors shape is {self.vectors.shape}")
        n_rows = self.signature_size
        n_cols = self.vectors.shape[1]
        hyperplanes = sparse.csr_matrix(
            self.rng.choice([-1, 1], (n_rows, n_cols))
        )
        products = self.vectors.dot(hyperplanes.transpose()).toarray()
        sign = 1 + (products > 0) # need +1 for cols_to_int_multidim
        self.signature = sign

    def _all_candidates_to_all(self):
        """Assign all mentions as candidates to all other mentions. 
        For edge cases where no single mention is longer than the shingle size.
        """
        n_mentions = self.vectors.shape[0]
        self.candidates = [set(range(n_mentions)) for _ in range(n_mentions)]

    def get_candidates(self):
        """Extract similar mentions from signature.

        For each mention, extract similar mentions based on whether part 
        of their signatures overlap.

        :return: Index of mentions that are similar to each other.
            A list of the candidate set of similar mentions.
        :rtype: list
        """
        logging.debug("getting candidates...")
        if self.vectors.shape[0] == 1:
            candidates = [set()]
            candidates[0].add(0)
        else:
            candidates = [set() for _ in range(self.vectors.shape[0])]

            bands = signature_to_3d_bands(self.signature, n_bands=self.n_bands, band_length=self.band_length)
            buckets_by_band = group_unique_indices(bands)
            groups = [tuple(i) for i in itertools.chain.from_iterable(buckets_by_band)] # flatten group; use tuple for applying set()
            groups = set(groups) # we only need the unique groups 

            for group in groups:
                for i in group:
                    candidates[i].update(group)

            [candidates[i].discard(i) for i in range(len(candidates))]
        self.candidates = candidates

    def cluster(self): 
        """End-to-end locality-sensitive hashing.

        Cluster mentions together based on their similarity. 

        :return: Index of mentions that are similar to each other.
            A list of the candidate set of similar mentions.
        :rtype: list
        """
        start = time.time()
        self._build_vocab()
        self.encode_binary()

        logging.debug("making signature")
        if self.vectors.shape[1] == 0: # no signature possible b/c no mention is longer than the shingle size.
            self._all_candidates_to_all()
        else:
            self.make_signature()
            self.get_candidates()
        self.time = time.time() - start 

    def summarise(self):
        "Summarise the time taken and output from clustering one LSH instance."
        sizes = [len(g) for g in self.candidates]
        print(f"took {self.time} seconds for {len(self.candidates)} mentions")
        print(f"average, min, max cluster size: {round(sum(sizes)/len(sizes),2)}, {min(sizes)}, {max(sizes)}")

    def efficiency_gain_comparisons(self):
        """
        Compare number of comparisons made for coreference search with option 
        "lsh" and option "all". Useful for understanding time complexity, 
        and to assess whether number of comparisons is meaningfully reduced.
        """
        sizes = [len(g) for g in self.candidates]
        runtime_all = len(self.candidates) * len(self.candidates)
        runtime_lsh = len(self.candidates) * (sum(sizes)/len(sizes))
        print(f"LSH makes fraction {round(runtime_lsh/runtime_all, 2)} of comparisons relative to option all.")
