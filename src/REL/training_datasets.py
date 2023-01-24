import os
import pickle
import pdb 
from REL.lsh import LSHRandomProjections
import logging

class TrainingEvaluationDatasets:
    """
    Class responsible for loading training/evaluation datasets for local ED.
    
    Reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, base_url, wiki_version, search_corefs="all"):
        """
        Argument search_corefs: One of 'all' (default), 'lsh', 'off'. 
            If 'off', no coreference search is done.
            Otherwise the arguments are passed to the argument `search_corefs_in` in `with_coref`.
        """
        self.person_names = self.__load_person_names(
            os.path.join(base_url, "generic/p_e_m_data/persons.txt")
        )
        self.base_url = os.path.join(base_url, wiki_version)
        assert search_corefs in ['all', 'lsh', 'off']
        self.search_corefs = search_corefs

    def load(self):
        """
        Loads respective datasets and processes coreferences.

        :return: Returns training/evaluation datasets.
        """
        datasets = {}
        for ds in [
            "aida_train",
            "aida_testA",
            "aida_testB",
            "wned-ace2004",
            "wned-aquaint",
            "wned-clueweb",
            "wned-msnbc",
            "wned-wikipedia",
        ]:

            print("Loading {}".format(ds))
            datasets[ds] = self.__read_pickle_file(
                os.path.join(self.base_url, "generated/test_train_data/", f"{ds}.pkl")
            )

            if ds == "wned-wikipedia":
                if "Jiří_Třanovský" in datasets[ds]:
                    del datasets[ds]["Jiří_Třanovský"]
                if "Jiří_Třanovský Jiří_Třanovský" in datasets[ds]:
                    del datasets[ds]["Jiří_Třanovský Jiří_Třanovský"]

            if self.search_corefs != "off":
                self.with_coref(datasets[ds], search_corefs_in=self.search_corefs)

        return datasets

    def __read_pickle_file(self, path):
        """
        Reads pickle file.

        :return: Dataset
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def __load_person_names(self, path):
        """
        Loads person names to find coreferences.

        :return: set of names.
        """

        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                data.append(line.strip().replace(" ", "_"))
        return set(data)

    def __find_coref(self, ment, mentlist):
        """
        Attempts to find coreferences

        :return: coreferences
        """

        cur_m = ment["mention"].lower()
        coref = []
        for m in mentlist:
            if (
                len(m["candidates"]) == 0
                or m["candidates"][0][0] not in self.person_names
            ):
                continue

            mention = m["mention"].lower()
            if mention == cur_m:
                continue
            start_pos = mention.find(cur_m)
            if start_pos == -1:
                continue

            end_pos = start_pos + len(cur_m) - 1
            if (start_pos == 0 or mention[start_pos - 1] == " ") and (
                end_pos == len(mention) - 1 or mention[end_pos + 1] == " "
            ):
                coref.append(m)

        return coref

    def with_coref(self, dataset, search_corefs_in="all"): # TODO: need to update the calls to with_coref
        """
        Check if there are coreferences in the given dataset. Use LSH for dimensionality reduction.

        search_corefs_in: either of 'lsh' or all 'all'. 
        If 'all', search for coreferences among all mentions in document. This is what REL currently does by default.
        If 'lsh', search for coreferences among a pre-selected set of candidates. The set is calculated with LSH.

        :return: dataset
        """
        print(f"with_coref() is called with search_corefs_in={search_corefs_in}.")
        assert search_corefs_in in ['lsh', 'all']
        for data_name, content in dataset.items():
            if len(content) == 0:
                pass 
            else:
                if search_corefs_in == 'lsh':
                    input_mentions = [m["mention"] for m in content]
                    # lsh_corefs = LSHRandomProjections(mentions=input_mentions, shingle_size=2, signature_size=800, band_length=10) # TODO: set optimal parameters here 
                    lsh_corefs = LSHRandomProjections(
                        mentions=input_mentions,
                        shingle_size=2,
                        n_bands=80,
                        band_length=10
                    )
                    lsh_corefs.cluster()
                    assert len(content) == len(lsh_corefs.candidates)
                    # lsh_corefs.candidates are the input for below. indices refer to index in input_mentions
                for idx_mention, cur_m in enumerate(content):
                    if search_corefs_in == "lsh":
                        idx_candidates = list(lsh_corefs.candidates[idx_mention]) # lsh returns the indices of the candidate coreferences
                        candidates = [content[i] for i in idx_candidates]
                    elif search_corefs_in == "all":
                        candidates = content
                    coref = self.__find_coref(cur_m, candidates)
                    if coref is not None and len(coref) > 0:
                        cur_cands = {}
                        for m in coref:
                            for c, p in m["candidates"]:
                                cur_cands[c] = cur_cands.get(c, 0) + p
                        for c in cur_cands.keys():
                            cur_cands[c] /= len(coref)
                        cur_m["candidates"] = sorted(
                            list(cur_cands.items()), key=lambda x: x[1]
                        )[::-1]
                        cur_m["is_coref"] = 1
                    else:
                        cur_m["is_coref"] = 0
