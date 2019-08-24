#!/bin/bash
echo Downloading $1
wget -P $data_dir http://nlp.cs.washington.edu/pair2vec/$1.tar.gz
tar xvzf $data_dir/$1.tar.gz -C $data_dir
rm $data_dir/$1.tar.gz
