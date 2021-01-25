# VNN-COMP 2021 

This repo contains the networks and benchmarks used for VNN COMP 2021.

## Instructions for Tool Authors
version 1
prepared by Stanley Bak, Jan 25, 2020

To facilitate automatic evaluation on the cloud, we need your tool in a standard format along with some bash scripts to setup and run the tool. You should put everything (your tool code and scripts) in a git repo that can be publically accessed. You then provide us with the git commit and we run everything. Note that your scripts can be used to download other resources from the web.

### Workflow
Networks will be provided in an .onnx file format that conforms to the VNNLIB standard. Properties are provided in a .vnnlib file. The .onnx and .vnnlib files will be downloaded to the home folder by cloning the following git repo: https://github.com/stanleybak/vnncomp2021 . For example, the test network / property will be in ~/vnncomp2021/benchmarks/test/test.onnx and ~/vnncomp2020/test/test.vnnlib

Your tool's .zip file will be copied to a fresh cloud instance running Ubuntu 20.04 LTS and unzipped in the home folder. We will then run an 'install_tool.sh' script to install your tool, and then it will launch 'setup_benchmark.sh' followed by 'run_benchmark.sh' many times, for each benchmark, which should create a result file.

 
### Scripts (should be in the top level of your git repo file)

* install_tool.sh: takes in single argument "v1", a version sting. This script is executed once to, for example, download any dependencies for your tool, compile any files, or setup licenses.

* setup_benchmark.sh: three arguments, first is "v1", second is path to the .onnx file and third is path to .vnnlib file. This script prepares the benchmark for evaluation (for example converting the onnx file to pytorch or reading in the vnnlib file to generate c++ source code for the specific property and compiling the code using gcc. You can also use this script to ensure that the system is in a good state to measure the next benchmark (for example, there are no zombie processes from previous runs executing and the GPU is available for use). This script should not do any analysis.

* run_benchmark.sh: five arguments, first is "v1", second is path to the .onnx file and third is path to .vnnlib file, fourth is a path to the results file, fifth is a timeout in seconds. Your script will be killed if it exceeds the timeone by too much, but sometimes gracefully quitting is better if you want to release resources cleanly like GPUs. The results file should be created after the script is run and contains two values "<result> <runtime>". The <result> can be sat, unsat, error, timeout, or other. The runtime is a floating point number that is the number of seconds your tool ran. We'll also meausure runtimes externally.

### Example
