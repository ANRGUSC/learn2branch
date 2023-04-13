#!/bin/bash

# Define the list of Python scripts to run

python 01_generate_instances.py makespan
python 02_generate_dataset.py makespan

for i in {0..2}
do
    python 03_train_gcnn.py makespan -m baseline -s $i
    python 03_train_gcnn.py makespan -m mean_convolution -s $i
    python 03_train_gcnn.py makespan -m no_prenorm -s $i
done

python 04_test.py makespan
python 05_evaluate.py makespan


echo "All scripts have been run"