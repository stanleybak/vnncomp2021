# VNN-COMP 2021 

This repository contains the networks and benchmarks used for the 2nd International Verification of Neural Networks Competition (VNN-COMP'21), as well as scripts for running the competition. The issues contains discussion about rules and benchmark proposals.

The VNN-COMP'21 website is here:  https://sites.google.com/view/vnn2021 

The repository for last year's (2020) competition is here: https://github.com/verivital/vnn-comp

## Instructions for Tool Authors
To facilitate automatic evaluation on the cloud, we need your tool in a standard format along with some bash scripts to setup and run the tool. Note that your scripts can be used to download resources from the web such as cloning git repositories. For future-proofing, cloning a specific commit or tag is the preferred method of downloading the tool.

### Workflow
Networks will be provided in an .onnx file format that conforms to the VNNLIB standard. Properties are provided in a .vnnlib file. The .onnx and .vnnlib files will be downloaded to the home folder by cloning the following git repo: https://github.com/stanleybak/vnncomp2021. For example, the test network / property will be in ~/vnncomp2021/benchmarks/test/test.onnx and ~/vnncomp2020/test/test.vnnlib.

Each tool author will be given access to an amazon cloud instance running Ubuntu 20.04 LTS, where they will download any required licenses for their tool. 

The tool's scripts will then be copied to the home folder. The VNNCOMP organizers will then run an 'install_tool.sh' script to install the tool, and then, for each benchmark instance, scipts will call 'prepare_instance.sh' followed by 'run_instance.sh', which should produce a results file.


### Scripts

Three scripts should be provided for each tool.

* `install_tool.sh`: takes in single argument "v1", a version string. This script is executed once to, for example, download any dependencies for your tool, compile any files, or setup any required licenses (if it can be automated). Note that some licences cannot be automatically retrived, so that the tool authors will be be responsible for a manual step prior to running any scripts to get the licenses.

* `prepare_instance.sh`: four arguments, first is "v1", second is a benchmark identifier string such as "acasxu", third is path to the .onnx file and fourth is path to .vnnlib file. This script prepares the benchmark for evaluation (for example converting the onnx file to pytorch or reading in the vnnlib file to generate c++ source code for the specific property and compiling the code using gcc. You can also use this script to ensure that the system is in a good state to measure the next benchmark (for example, there are no zombie processes from previous runs executing and the GPU is available for use). This script should not do any analysis. The benchmark name is provided, as per benchmark settings are permitted (per instance settings are not, so do NOT use the onnx filename vnnlib filename to customize the verification tool settings).

* `run_instance.sh`: six arguments, first is "v1", second is a benchmark identifier string such as "acasxu", third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds. Your script will be killed if it exceeds the timeone by too much, but sometimes gracefully quitting is better if you want to release resources cleanly like GPUs. The results file should be created after the script is run and is a simple text file containing one word on a single line: holds, violated, timeout, error, or unknown.

