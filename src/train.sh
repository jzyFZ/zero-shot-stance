#!/usr/bin/env bash


train_data=data/VAST/vast_train.csv
dev_data=data/VAST/vast_dev.csv

echo "training model with early stopping and $3 warm-up epochs"
python train_model.py -c $1 -t ${train_data} -d ${dev_data} -e 1 -w $2 -k $3
