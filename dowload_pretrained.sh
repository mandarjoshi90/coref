#!/bin/bash
if [ $1 = 'bert_base' ]
then
    model=english_sl128_ff3000_blr1e-05_tlr0.0002
elif [ $1 = 'bert_large' ]
then
	model=large_english_sl384_ff3000_blr1e-05_tlr0.0002
elif [ $1 = 'spanbert_base' ]
then
	model=base_big_batch_no_nsp_pair_sl384_blr2e-05_tlr0.0001
else
	model=small_batch_no_nsp_pair_sl512_blr1e-05_tlr0.0003
fi
echo Downloading $1 -- $model
wget -P $data_dir http://nlp.cs.washington.edu/pair2vec/$model.tar.gz
tar xvzf $data_dir/$model.tar.gz -C $data_dir
rm $data_dir/$model.tar.gz
