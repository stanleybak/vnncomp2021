#!/bin/bash -ve
# setup variables for aws testing


# AWS connection server username@ip
export SERVER=ubuntu@ec2-18-191-230-227.us-east-2.compute.amazonaws.com

# name of tool to use
export TOOL=ERAN

# path to tool script file (inside vnnlib repo)
export TOOL_SCRIPT="/home/stan/repositories/vnncomp2021/tools/${TOOL}.sh"

# AWS EC2 key for ssh access
export PEM="/home/stan/.ssh/vnn-comp-bak.pem"

# config file for email credentials for status notification
export SSMTP_CONF_PATH="/home/stan/ssmtp.conf"
