
BASE_URL="$1"

DATASETS=("aida_testB")
DOCSIZES=(50 500)
COREF_OPTIONS=("all" "off" "lsh")


echo $DATASETS


echo "--Running efficiency tests by data set, n_docs and coref option--"

# do profiling and checking predictions in one 
for size in ${DOCSIZES[@]}; do
    for ds in ${DATASETS[@]}; do
        for option in ${COREF_OPTIONS[@]}; do
            echo $ds, echo $size, echo $option 
            python scripts/efficiency_test.py \
                --url "$BASE_URL" \
                --profile \
                --n_docs $size \
                --name_dataset "$ds" \
                --search_corefs $option 
        done
    done 
done 

# echo "--Scaling number of mentions--"

# for ds in ${datasets[@]}; do
#     echo $ds
#     python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "all"
#     python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "lsh"
#     python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "off"
# done 


echo "Done."



