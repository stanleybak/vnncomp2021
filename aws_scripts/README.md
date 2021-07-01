Seed:
# seed taken from ethereum block 12735571 hash on mined on June 30, 2021 at 9:18 AM EDT
SEED_HEX="95f9d11b93aa0745e51dc13fc00a373f4c34c534f39ef87df6f3f0818db16203"
SEED=$(python3 -c "print(int('${SEED_HEX}', 16) % 1000000000)")

1. start (CPu or GPU) AWS ec2 server and get usename@ip address from connect menu. put this into the step0_config.sh script, along with the tool name and path to pem file to connect to the ec2 instance
2. run step1_setup.sh (copies files)
3. run step2_install.sh (automatic installation) / do manual installation steps (read the tool's README)
4. run step3_test.sh (run tool on test benchmark). make sure no errors and test_sat is sat or unknown
5. run step4_run.sh. This takes a while and will run in background. This script will auto shutdown the server when done (or after ~66 hours), as well as send me an email with the results.


