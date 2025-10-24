#!/bin/bash
# Script for splitting the dataset folder to train, val, test sets.

# Shuffles the data too.

INPUT="/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/patches"
OUTPUT="/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"

if [ -d "$OUTPUT" ] && [ -z "$(ls -A "$OUTPUT")" ]; then
	echo "directory empty"
	splitfolders --output $OUTPUT --ratio .7 .2 .1 --move -- $INPUT
else
	echo "directory not empty..."
	echo "emptying directory"
	rm -rf "$OUTPUT"/*
	splitfolders --output $OUTPUT --ratio .7 .2 .1 --move -- $INPUT 	
fi
 

