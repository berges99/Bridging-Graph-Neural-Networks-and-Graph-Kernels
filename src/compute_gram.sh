#!/bin/bash

# Specify all desired datasets
DATASETS=("IMDBMULTI")


# Perform small grid-search for the parameters of the GNTK
for df in "${DATASETS[@]}"; do
	# Number of blocks
	for s in {1..3}; do
		# Number of layers
		for l in {1..3}; do
			# Readout operation
			for r in "sum" "jkn"; do
				# Scaling factor
				for scale in "uniform" "degree"; do
					# Run the python script
					python compute_gntk.py --dataset $df -S $s -L $l -ro $r -scale $scale
done; done; done; done; done;