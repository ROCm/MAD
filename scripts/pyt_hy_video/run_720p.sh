#!/bin/bash

bash run.sh -h 720
bash run.sh -h 960
bash run.sh -h 1280

python merge_csv.py
