import numpy as np
import requests
import argparse
import pickle
import time 
from functools import wraps


from REL.training_datasets import TrainingEvaluationDatasets

np.random.seed(seed=42)

def time_runs(func):
    "Time the execution of multiple calls to `func`."
    @wraps(func)
    def t(*args, **kwargs):
        i = 0 
        n_runs = kwargs["n_runs"]
        timing_md = []
        while i < n_runs:
            print(f"run {i} out of {n_runs}")
            start = time.time()
            func(*args)
            time_run = time.time() - start
            timing_md.append(time_run)
            i += 1
        return timing_md
    return t


parser = argparse.ArgumentParser()
parser.add_argument("--use_corefs", action="store_true", help="use function with_coref()?", default=False)
parser.add_argument("--n_runs", type=int, help="iteration number for timing MD and ED steps.", default=None)
args = parser.parse_args()
print(f"args.use_corefs is {args.use_corefs}")


base_url = "/home/flavio/projects/rel20/data"
wiki_version = "wiki_2019"
datasets = TrainingEvaluationDatasets(base_url, wiki_version, args.use_corefs).load()["aida_testB"] 
    # datasets are loaded here, then processed and stored in docs, which is then used to check the efficiency


# random_docs = np.random.choice(list(datasets.keys()), 50)

server = False
docs = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == 50:
        print("length docs is 50.")
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]
        # Demo script that can be used to query the API.
        if server:
            myjson = {
                "text": text,
                "spans": [
                    # {"start": 41, "length": 16}
                ],
            }
            print("----------------------------")
            print(i, "Input API:")
            print(myjson)

            print("Output API:")
            print(requests.post("http://192.168.178.11:1235", json=myjson).json())
            print("----------------------------")


# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not server:
    #from time import time

    import flair
    import torch
    from flair.models import SequenceTagger

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.mention_detection import MentionDetection

    # base_url = "C:/Users/mickv/desktop/data_back/" # why is this defined again here?

    flair.device = torch.device("cpu")

    mention_detection = MentionDetection(base_url, wiki_version)

    # Alternatively use Flair NER tagger.
    tagger_ner = SequenceTagger.load("ner-fast")

    # run it once more for next step
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)

    if args.n_runs is not None:
        print("Timing MD...")
        @time_runs
        def time_find_mentions(docs, tagger_ner):
            mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)

        timing_md = time_find_mentions(docs, tagger_ner, n_runs=args.n_runs)


    # 3. Load model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    model = EntityDisambiguation(base_url, wiki_version, config, use_corefs=args.use_corefs)
        # model.coref is a training data set
        # model.coref has method with_coref
        # compare the training data sets when using corefs and when not
        # note that the data are loaded elsewhere! so not sure this is the right place to add the option? 

    # 4. Entity disambiguation.
    predictions, timing = model.predict(mentions_dataset)

    if args.n_runs is not None:
        print("Timing ED...")
        @time_runs
        def time_predict(mentions):
            predictions, timing = model.predict(mentions)

        timing_ed = time_predict(mentions_dataset, n_runs=args.n_runs)

    # 5. Store output
    if args.n_runs is not None:
        results = {
            "MD": timing_md,
            "ED": timing_ed,
        }
        fn = "timing"
    else:
        results = {
            "predictions": predictions,
            "timing": timing
        }
        fn = "output"

    
    fn = f"{base_url}/efficiency_test/{fn}"
    if not args.use_corefs:
        fn = f"{fn}_nocoref"

    with open(f"{fn}.pickle", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    
