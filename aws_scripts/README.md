## Set up tools

This folder containts scripts to help with evaluation on AWS.


Instructions:

1. update step0_config.sh TOOL_SCRIPT, PEM, and SSMTP_CONF_PATH variables based on your local system paths (one time)

2. Update tool scripts based on tool repo / commit information (see vnncomp2021/tools for examples)

3. When you spawn an EC2 instance, put the server information into the tool script SERVER variable.

3. Run one-by-one, checking the output at each step:
./step1_setup.sh TOOL_NAME
./step2_install.sh TOOL_NAME
./step3_test.sh TOOL_NAME
./step4_run.sh TOOL_NAME

Step 2 and 3 will output .txt files in the local directory of stdout/stderr, useful for debugging problems

Step 4 will run on AWS EC2 in tmux, so that disconnecting from ssh keeps the process alive.

## Get gurobi license

Getting Gurobi license requires the machine connected to the university network. But once an instance is connected to a VPN, you will lose your ssh connection. So we automated this process in vpn_gurobi.sh. This script will connect to your university VPN and run `grbgetkey` command and then disconnect, such that the SSH connection is preserved.

To get a gurobi license, run the following command on the AWS instance:
```bash
# If your VPN has no GROUP option, please remove it from vpn_gurobi.sh. 
# GUROBI_USER is the system user that will use the gurobi license. The license is only valid for one user. For a AWS ubuntu instance, this is usually "ubuntu". 
# GUROBI_KEY is from https://www.gurobi.com/downloads/free-academic-license/. It looks like: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX. The key expires soon, please get the key before running the script. You can get unlimited number of keys.

sudo ./vpn_gurobi.sh SERVER GROUP USERNAME PASSWARD GUROBI_USER GUROBI_KEY
```

An example is:
```bash
sudo ./vpn_gurobi.sh vpn.cmu.edu "Full VPN" twei2 mypassword ubuntu 882d4d74-e015-11eb-8ce9-0242acXXXXXX
```