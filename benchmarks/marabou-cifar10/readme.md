This folder contains targeted robustness query for three convolutional ReLU networks
trained on the CIFAR10 dataset.
The architecture of the three networks can be found in `info.txt`

The input image is normalized between 0-1.
The target label is set to be `(correct label + 1 % 10)`

To generate the benchmarks with a particular random seed, run:
   `./generate_benchmarks.sh {seed}`
which generates the vnn lib files, the csv file marabou-cifar10_instances.csv
