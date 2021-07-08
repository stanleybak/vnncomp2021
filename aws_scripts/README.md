## Set up tools

scripts to help with evaluation on AWS

1. start (CPu or GPU) AWS ec2 server and get usename@ip address from connect menu. put this into the step0_config.sh script, along with the tool name and path to pem file to connect to the ec2 instance
2. run step1_setup.sh (copies files)
3. run step2_install.sh (automatic installation) / do manual installation steps (read the tool's README)
4. run step3_test.sh (run tool on test benchmark). make sure no errors and test_sat is sat or unknown
5. run step4_run.sh. This takes a while and will run in background. This script will auto shutdown the server when done (or after ~66 hours), as well as send me an email with the results.

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
