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
