import numpy as np
import requests
import argparse
import pickle
import logging 

from REL.training_datasets import TrainingEvaluationDatasets

np.random.seed(seed=42)

parser = argparse.ArgumentParser()
# parser.add_argument( 
#     "--no_corefs",
#     action="store_true",
#     help="use function with_coref()?", 
#     default=False)
parser.add_argument(
    '--search_corefs',
    type=str,
    choices=['all', 'lsh', 'off'],
    default='all',
    help="Setting for search_corefs in Entity Disambiguation."
)

parser.add_argument(
    "--profile",
    action="store_true",
    default=False,
    help="Profile the disambiguation step."
    )
parser.add_argument(
    "--scale_mentions",
    action="store_true", 
    default=False,
    help="""Stack mentions in each dataset and time the disambiguation step by document. 
            This is to assess the time complexity of the program."""
    )
parser.add_argument(
    "--name_dataset",
    type=str,
    default="aida_testB",
    help="Name of the training dataset to be used"
)
parser.add_argument(
    "--n_docs",
    type=int,
    default=50,
    help="Number of documents to be processed."
)
logging.basicConfig(level=logging.INFO) # do not print to file 


# helper function to profile a call and save the timing in a pd dataframe 
def profile_to_df(call):
    cProfile.run(call, filename="temp.txt")
    st = pstats.Stats("temp.txt")

    keys_from_k = ['file', 'line', 'fn']
    keys_from_v = ['cc', 'ncalls', 'tottime', 'cumtime', 'callers']
    data = {k: [] for k in keys_from_k + keys_from_v}

    s = st.stats

    for k in s.keys():
        for i, kk in enumerate(keys_from_k):
            data[kk].append(k[i])

        for i, kk in enumerate(keys_from_v):
            data[kk].append(s[k][i])

    df = pd.DataFrame(data)
    os.remove('temp.txt')
    return df



# TODO:
# make log files!?
# adjust folder structure on computer and in script 

args = parser.parse_args()
print(f"args.search_corefs is {args.search_corefs}")

if args.profile:
    import cProfile 
    import pandas as pd 
    import pstats 
    import os 


base_url = "/home/flavio/projects/rel20/data"
wiki_version = "wiki_2019"
datasets = TrainingEvaluationDatasets(base_url, wiki_version, args.search_corefs).load()[args.name_dataset] 
save_data_to = f"{base_url}/efficiency_test/" # save all recorded in this directory 

# random_docs = np.random.choice(list(datasets.keys()), 50)

server = False
docs = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == args.n_docs:
        print(f"length docs is {args.n_docs}.")
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
    from time import time

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

    start = time()
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner) # TODO: here corefs have an impact! check how.
        # but what we do in the mention detection here has no impact on what we below in ED. 
        # so would we expect an effect here, or only below?
    print("MD took: {}".format(time() - start))

    # 3. Load model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    model = EntityDisambiguation(base_url, wiki_version, config, search_corefs=args.search_corefs) 
        # model.coref is a training data set
        # model.coref has method with_coref
        # compare the training data sets when using corefs and when not
        # note that the data are loaded elsewhere! so not sure this is the right place to add the option? 

    # 4. Entity disambiguation.
    start = time()
    predictions, timing = model.predict(mentions_dataset)
    print("ED took: {}".format(time() - start))

    output = {
        "mentions": mentions_dataset,
        "predictions": predictions,
        "timing": timing
    }
    
    filename = f"{save_data_to}predictions/{args.name_dataset}_{args.n_docs}_{args.search_corefs}"
    # if args.no_corefs:
    #     filename = f"{filename}_nocoref"

    with open(f"{filename}.pickle", "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)        

    # ## 4.b Profile disambiguation
    if args.profile:
        print("Profiling disambiguation")
        filename = f"{save_data_to}profile/{args.name_dataset}_{args.n_docs}_{args.search_corefs}"
        # if args.no_corefs:
        #     filename = f"{filename}_nocoref"

        df_stats = profile_to_df(call="model.predict(mentions_dataset)")
        # cProfile.run("model.predict(mentions_dataset)", filename="temp.txt")
        # st = pstats.Stats("temp.txt")

        # keys_from_k = ['file', 'line', 'fn']
        # keys_from_v = ['cc', 'ncalls', 'tottime', 'cumtime', 'callers']
        # data = {k: [] for k in keys_from_k + keys_from_v}

        # s = st.stats

        # for k in s.keys():
        #     for i, kk in enumerate(keys_from_k):
        #         data[kk].append(k[i])

        #     for i, kk in enumerate(keys_from_v):
        #         data[kk].append(s[k][i])

        # df_stats = pd.DataFrame(data)
        # os.remove('temp.txt')

        df_stats.to_csv(f"{filename}.csv", index=False)


    # ## 4.c time disambiguation by document, vary number of mentions 
    if args.scale_mentions:
        print("Scaling the mentions per document")
        mentions_dataset_scaled = {}

        for k, data in mentions_dataset.items():
            mentions_dataset_scaled[k] = data # add the baseline data as in mentions_dataset
            for f in [5, 10, 50, 100]:
                d = data * f 
                key = f"{k}_{f}"
                mentions_dataset_scaled[key] = d

        print("Timing disambiguation per document")
        timing_by_dataset = {}
        for name, mentions in mentions_dataset_scaled.items():
            print(f"predicting for dataset {name}")
            tempdict = {name: mentions} # format so that model.predict() works 
            start = time()
            predictions, timing = model.predict(tempdict)
            t = time() - start

            timing_by_dataset[name] = {
                "n_mentions": len(mentions),
                "time": t
            }
            
            if args.profile:
                df_profile = profile_to_df(call="model.predict(tempdict)") 
                timing_by_dataset[name]['profile'] = df_profile

        
        # save timing by dataet
        filename = f"{save_data_to}n_mentions_time/{args.name_dataset}_{args.search_corefs}"
        # if args.no_corefs:
        #     filename = f"{filename}_nocoref"
        
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(timing_by_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)



    
