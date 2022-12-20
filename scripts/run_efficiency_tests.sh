


datasets=("aida_testB")

docsizes=(50 500)


echo $datasets

# do profiling and checking predictions in one 
for size in ${docsizes[@]}; do
    for ds in ${datasets[@]}; do
        echo $ds, echo $size
        python scripts/efficiency_test.py --profile --n_docs $size --name_dataset "$ds"
        python scripts/efficiency_test.py --profile --n_docs $size --name_dataset "$ds" --no_corefs
    done 
done 


# for ds in ${datasets[@]}; do
#     echo $ds
#     python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile 
#     python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --no_corefs
# done 






