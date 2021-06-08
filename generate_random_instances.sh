#!/bin/bash -e
# generate random benchmarks based on seed

SEED=0

if [ "$#" -eq 1 ]; then
    SEED=$1
fi


pushd benchmarks/mnistfc
#pip3 install torch==1.8.1
python3 generate_properties.py $SEED
popd

echo "Successfully generated all random instances."
