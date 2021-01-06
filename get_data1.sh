#!/bin/bash

ROOT="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets"
DATASETS=( "COLLAB" "IMDB-BINARY" "IMDB-MULTI" "MUTAG" )

echo "Downloading the following datasets: ${DATASETS[*]}"

for df in "${DATASETS[@]}"
do
	dir="data/${df}"
	# Make directory if it does not exist
	if [ ! -d "${dir}" ]
	then
		mkdir "$dir"
	fi
	# Download zip data inside the directory
	wget -O "${dir}/${df}.zip" "${ROOT}/${df}.zip"
	# Unzip the content
	unzip -d "${dir}" "${dir}/${df}.zip"
done

echo "Download successful!"