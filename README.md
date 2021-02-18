# VNN-COMP 2021 

This repo contains the networks and benchmarks used for VNN COMP 2021.

## Instructions for Tool Authors
version 2
updated Feb 18, 2020

To facilitate automatic evaluation on the cloud, we need your tool in a standard format along with some bash scripts to setup and run the tool. Note that your scripts can be used to download resources from the web such as cloning git repositories. For future-proofing, cloning a specific commit or tag is the preferred method of downloading the tool.

### Workflow
Networks will be provided in an .onnx file format that conforms to the VNNLIB standard. Properties are provided in a .vnnlib file. The .onnx and .vnnlib files will be downloaded to the home folder by cloning the following git repo: https://github.com/stanleybak/vnncomp2021. For example, the test network / property will be in ~/vnncomp2021/benchmarks/test/test.onnx and ~/vnncomp2020/test/test.vnnlib.

Each tool author will be given access to an amazon cloud instance running Ubuntu 20.04 LTS, where they will download any required licenses for their tool. 

The tool's scripts (provided in a .zip file) will then be copied to the home folder and unzipped. The organizers will then run an 'install_tool.sh' script to install the tool, and then, for each benchmark instance, scipts will call 'setup_benchmark.sh' followed by 'run_benchmark.sh', which should produce a results file.

 
### Scripts (should be in the top level of your git repo file)

* install_tool.sh: takes in single argument "v2", a version sting. This script is executed once to, for example, download any dependencies for your tool, compile any files, or setup licenses.

* setup_benchmark.sh: four arguments, first is "v2", second is a benchmark category identifier string such as "acasxu", third is path to the .onnx file and fourth is path to .vnnlib file. This script prepares the benchmark for evaluation (for example converting the onnx file to pytorch or reading in the vnnlib file to generate c++ source code for the specific property and compiling the code using gcc. You can also use this script to ensure that the system is in a good state to measure the next benchmark (for example, there are no zombie processes from previous runs executing and the GPU is available for use). This script should not do any analysis. The benchmark category is provided, as per category settings are permitted (per instance settings are not).

* run_benchmark.sh: six arguments, first is "v2", second is a benchmark category itentifier string such as "acasxu", third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds. Your script will be killed if it exceeds the timeone by too much, but sometimes gracefully quitting is better if you want to release resources cleanly like GPUs. The results file should be created after the script is run and is a simple text file containing two values "<result> <runtime>". The <result> can be sat, unsat, error, timeout, or other. The runtime is a floating point number that is the number of seconds your tool ran. We'll also meausure runtimes externally.

### Example
