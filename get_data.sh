#!/bin/bash

URL="http://pascal.inrialpes.fr/data2/dchen/data/graphs.zip"
DIR="data/raw"

# Download all the datasets
wget -O "${DIR}/graphs.zip" "${URL}"

# Unzip the datasets in the data directory
unzip -d "${DIR}" "${DIR}/graphs.zip"