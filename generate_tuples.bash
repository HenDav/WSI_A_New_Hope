#!/bin/bash

cd /home/royve/Github/WSI_MIL

export PYTHONPATH=.

/home/royve/miniconda3/envs/wsi-mil/bin/python apps/generate_tuples.py \ 
--inner-radius 2 \
--outer-radius 10 \
--test-fold 0 \
--train True \
--tile-size 256 \
--desired-magnification 10 \
--metadata-file-path None \ 
--metadata-enhancement-dir-path /home/royve/enhancement \
--datasets-base-dir-path /mnt/gipmed_new/Data \
--dataset-ids TCGA \
--minimal-tiles-count 10 \
--folds-count 6 \
--tuples-count 1000 \
--negative-examples-count 2 \
--tuples-dir-path /home/royve/output \
--num-workers 5