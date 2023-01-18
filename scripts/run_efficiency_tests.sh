


datasets=("aida_testB")

docsizes=(50 500)


echo $datasets


echo "--Running efficiency tests by data set and n_docs--"

# do profiling and checking predictions in one 
# for size in ${docsizes[@]}; do
#     for ds in ${datasets[@]}; do
#         echo $ds, echo $size
#         python scripts/efficiency_test.py --profile --n_docs $size --name_dataset "$ds" --search_corefs "all"
#         python scripts/efficiency_test.py --profile --n_docs $size --name_dataset "$ds" --search_corefs "lsh"
#         python scripts/efficiency_test.py --profile --n_docs $size --name_dataset "$ds" --search_corefs "off"
#     done 
# done 

echo "--Scaling number of mentions--"

for ds in ${datasets[@]}; do
    echo $ds
    python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "all"
    python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "lsh"
    python scripts/efficiency_test.py --name_dataset "$ds" --scale_mentions --profile --search_corefs "off"
done 


echo "Done."



