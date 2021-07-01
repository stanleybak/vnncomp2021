scripts to help with evaluation on AWS

1. start (CPu or GPU) AWS ec2 server and get usename@ip address from connect menu. put this into the step0_config.sh script, along with the tool name and path to pem file to connect to the ec2 instance
2. run step1_setup.sh (copies files)
3. run step2_install.sh (automatic installation) / do manual installation steps (read the tool's README)
4. run step3_test.sh (run tool on test benchmark). make sure no errors and test_sat is sat or unknown
5. run step4_run.sh. This takes a while and will run in background. This script will auto shutdown the server when done (or after ~66 hours), as well as send me an email with the results.


