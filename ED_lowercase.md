ED with lower-case data 
========================

This note explains what I think needs to be adapted to train an ED model for REL with lower-case data. The steps are based on the [tutorial for a new Wikipedia corpus](https://github.com/informagi/REL/blob/f4c471a18f0d2124bd04cf3e2f2fbffcb72e16fd/docs/tutorials/deploy_REL_new_wiki.md#training-wikipedia2vec-embeddings), and partly on the paper.

# Overview
1. Creating a folder structure
    - No changes
    - But: there are duplicated rows in the database in the `lower` column. Currently the package just loads the first entry in this case. I am not sure this is the intended use, but don't think there is a way to avoid it. Maybe a warning could be issued?
2. Embeddings
    - run `Wikipedia2Vec` with the lower case option
    - Then the embeddings are stored in the db (see "Storing Embeddings in DB").
    - once this is done, will the primary keys between the table still correspond -- this depends on `Wikipedia2Vec` outputs with the lower case option
3. Generating training, validation and test files
    - No direct change in the processing
    - See the file `generate_training_test`
        - it stores the data to `data/wiki_version/generated/test_train_data/`. The mentions are in capitals
            - Implication: no need to re-generate the data here, but for the training the keys of the dictionary need to be put into lower case 
4. Training your own Entity Disambiguation model
    - largely follow the existing instructions
    - uncase the keys in the method `TrainingEvaluationDatasets.load()` (from point 3 above).


# Details

## 1. Creating a folder structure
- data exist and are in right structure
- extracting a wikipedia dump
    - we just use the existing data 
- generate p(e|m) index (*--this will be used for the candidate selection*) as follows
    ```python
    wiki_yago_freq = WikipediaYagoFreq() # initialize
    wiki_yago_freq.compute_wiki() # p(e|m) index
    wiki_yago_freq.compute_custom() # Yago and combine with p(e|m)
    wiki_yago_freq.store() # write to database. database columns: (word, p_e_m, lower, freq)
    ```
- The key output from this step is the database table `wiki`. The database serves the package with the candidate mentions, which are queried in later steps.

**Does it matter whether we use lower-case data or not?**

*Do the queries work? / Do they return the same things with and without lowercase?*
- The query may work mechanically because we have a column `lower` and can search for the mention by referring to the `lower` column
- The problem could be that there are duplicates in the `lower` column:
    ```sql
    select count(*) from wiki; -- returns 23202365
    select count(distinct word) from wiki; -- returns 23202365
    select count(distinct lower) from wiki; -- returns 16011257
    ```
    - Example duplicates seem to be mostly different spelling of the same entity, ie `select * from wiki where lower = "cinco de mayo" ;` or `select * from wiki where lower = "nextstep" ;`
    - So, what is this table again for exactly? -- they key is **how** the database is used.
        - We look for `word` in the database -- that should work fine
        - We look for `lower` in the database. It seems that REL currently picks a random row in this case (`.fetchone`), which could be problematic because
                - The content of `freq` varies across rows with the same value of `lower` (see cinco de mayo example)
                - The content of `p_e_m` varies across rows with the same value of `lower` (see nextstep example)
        - This not only affects the proposed change for lowercase, but is already used at the moment
            - when running `efficiency_test.py`, it is called from `mention_detection.find_mentions()`. I suppose this is used to make the predictions
            - Here, the column `freq` is queried.
            - In [`mention_detection.MentionDetectionBase()`](https://github.com/informagi/REL/blob/d3e24ea67e003ad50e619f6c3012ee9522fffcea/src/REL/mention_detection_base.py) as a fallback when the capitalized mention is not found in the database. This is calld for instance in `mention_detection.find_mentions()`.
                - Because most entries in `lower` are still unique, this works in many cases. But for my tests, it could fail for instance for the mention "NWS"
        - Moreover, I suppose it is used for training, but have not verified 
        - In practice, this probably adds some noise to the data makes the prediction less precise. But not hugely so because the majority of entries in `lower` are still unique. 

- Here is how the database is queried (class `DB` in REL.db.base) -- it is used in `GenericLookup.wiki()`
```python
def lookup_wik(self, w, table_name, column):
    """
    Args:
        w: word to look up.
    Returns:
        embeddings for ``w``, if it exists.
        ``None``, otherwise.
    """
    if column == "lower": # so what happens here if the entries in lower are not unique?
        e = self.cursor.execute(
            "select word from {} where {} = :word".format(table_name, column),
            {"word": w},
        ).fetchone()
    else:
        e = self.cursor.execute(
            "select {} from {} where word = :word".format(column, table_name),
            {"word": w},
        ).fetchone()
    res = (
        e if e is None else json.loads(e[0].decode()) if column == "p_e_m" else e[0]
    )

    return res
```



*Do the calculated p_e_m scores depend on whether mentions are lowercased or not? (ignoring the above issue)*
- Check again how exactly the p_e_m scores are calculated 



*Some other notes*
- I instantiated a `wikipedia = Wikipedia()` instance. `wikipedia.wiki_id_name_map` is a dict with keys `[ent_name_to_id, ent_id_to_name]`. 
    - the keys are the named entities, eg "Alexander Aetolos" 


### Summary
Given the information so far, the idea seems to be to leave the Wikipedia data as-is and just use the fallback query with lower when using REL with lower-case data. 


## 2. Embeddings


### a) Wikipedia2Vec
- need the zipped data 
- `wikipedia2vec`
    - see the documentation of that package.  `wikipedia train` is the main function and has the option `--lowercase`/`--no-lowercase` and calls several scripts
    - in file `REL/scripts/preprocess.sh` 
        - add the option lower case (what is the default?) `wikipedia2vec build-dictionary dump_file dump_dict --min-entity-count 0` 
        - the function `build-mention-db` has an option `--case-sensitive`, so do we have to fix this as well?
            - --> how is it implemented in the main `train` function? 
            - the default [here](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/wikipedia2vec/cli.py#L64) seems to be False? is this also used for the current default in REL? can I see this somewhere?
    - the file `REL/scripts/train.sh` then uses the output from the previous file for training. 
        - I think there is nothing to be changed for the lowercase option 

**Thoughts, comments and questions**
- Is there a reason all this stuff is not part of the package? ie, `wikipedia2vec` is not in the package and needs pip install; when we extend and allow for the lower case option, do we want this to be an option in the whole package, or leave it as it is for now outside of it (but instead in a tutorial?)
    - Answer: See the [tutorial](https://github.com/informagi/REL/blob/f4c471a18f0d2124bd04cf3e2f2fbffcb72e16fd/docs/tutorials/deploy_REL_new_wiki.md#training-wikipedia2vec-embeddings)? "Some steps are outside the scope of REL"


### b) Store embeddings in DB 
- huge file -> store embeddings in DB 
- does this change when using lower case data?
    - how are embeddings calculated? the key question seems to be whether
        - the data are read with capitals 
        - whether the model is case-sensitive
    - the words at least are stored with capitals **<-- how is this table used, and where/how is it generated?**
        ```sql
        select *
        from (
            select *
                , substr(word, 1, 1) as first_letter
                , lower(substr(word, 1, 1)) as first_letter_lower  
            from wiki
        ) 
        where first_letter != first_letter_lower and first_letter_lower = "a"
        limit 10;
        ```
        - [here](https://github.com/informagi/REL/blob/main/docs/tutorials/deploy_REL_new_wiki.md) is an example of how the table is generated.
            - If I understand correctly, the classes `Wikipedia`, `WikipediaYagoFreq` 
    - but not sure there are any words in the  `embeddings` table 

## 3. Generating training, validation and test files
- instantiate class `GenTrainingTest`: it parses the raw training and test files that are in the generic/ folder
    - which of the listed data sets should be used?
- does this change when using lower case data?
    - presumably the training data need to change because they have labels for each entity, and entities with capital or lower case are different in the text?

## 4. Training your own Entity Disambiguation model

- We only need to change the input data; once this is properly set up the training and evaluation should work in the same way 

1. load training and evaluation data sets in folder `generated/`
    - uncase the keys in the method `TrainingEvaluationDatasets.load()`
2. define config dictionary
    - according to instructions
3. train or evaluate the model
    - I don't understand the syntax there 
4. train model 
5. obtain confidence scores 


---

How does the training work? -- schematic view of the `train()` method
- set up optimizer (`torch.optim`)
- datasets
    - `train_data_set`: `predict=False` *-- ?*
    - `dev_dataset`: `predict=True` *-- ?*
- for `epoch` in (0, 1, ... `n_epochs`)
    - for doc in (0, .... n_documents) *iterate over train_dataset that consists of documents. in each `epoch`, the order of the documents is randomized*
        - `self.model.train()` -- what does this do exactly?
        - `self.model.zero_grad()` (?) -- and this?
        - convert data items to pytorch inputs
        - `self.model.forward(); .loss(); .backward(); ` *--see the paper by Le and Titov*
        - print out the `epoch`, progress (as % of all documents) and the `loss`
    - print out `epoch`, total loss, average loss per document
    - after each `eval_after_n_epochs`, the current performance of the model is assessed and printed (the recall, precision)
- In sum
    - tune the length of the training with the parameter `n_epochs`
    - how does the number of epochs impact the performance of the model? why are more epochs better?
        - in 1 epoch, the full data set is run once through the network (possibly in batches). for each batch, we make a forward and backward pass through the network. 
    - it still unclear to me why we call `train()` and `zero_grad()` in each batch
    - one explanation: we evaluate the model at the current parameters sequentially for each batch and calculate the loss. If this is the case, where are the parameters updated? 
        - the parameters change within epochs across datasets (minibatch) *(what is a batch then??)*
        - where does it happen? maybe in the `optimizer.step()` function? 
- It would be good to store the printed output from one full training to see what is going on and how long it takes

