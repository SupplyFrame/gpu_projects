#!/bin/bash

label_file='labels.txt'
data_file='data.txt'
examples=`cat $label_file|wc -l`
layer_1_dim=`head -n1 $data_file|wc -w`

echo ./nn $examples $label_file $data_file 3 $layer_1_dim 4 3
./nn $examples $label_file $data_file 3 $layer_1_dim 4 3
