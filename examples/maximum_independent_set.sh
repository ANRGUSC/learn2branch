#!/bin/bash

# cd to script directory
cd "$(dirname "$0")"

cd ..
python 01_generate_instances.py indset
# Generate supervised learning datasets
python 02_generate_dataset.py indset -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py indset -m baseline -s $i
    python 03_train_competitor.py indset -m extratrees -s $i
    python 03_train_competitor.py indset -m svmrank -s $i
    python 03_train_competitor.py indset -m lambdamart -s $i
done
# Test
python 04_test.py indset
# Evaluation
python 05_evaluate.py indset

# cd to script directory
cd "$(dirname "$0")"