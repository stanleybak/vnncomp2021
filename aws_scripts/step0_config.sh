#!/bin/bash -e
# setup variables for aws testing
# assumes TOOL env variable was exported before running

# path to tool script file (inside vnnlib repo)
export TOOL_SCRIPT="/home/stan/repositories/vnncomp2021/tools/${TOOL}.sh"

# AWS EC2 key for ssh access
export PEM="/home/stan/.ssh/vnn-comp-bak.pem"

# config file for email credentials for status notification
export SSMTP_CONF_PATH="/home/stan/ssmtp.conf"





######################
# check if files exist

if [ ! -f $TOOL_SCRIPT ] 
then
    echo "Tool script does not exist: '$TOOL_SCRIPT'"
    exit 1
fi

SERVER=""
source $TOOL_SCRIPT

if [ ! -f $PEM ] 
then
    echo "PEM path does not exist: '$PEM'"
    exit 1
fi

if [ ! -f $SSMTP_CONF_PATH ] 
then
    echo "SSMTP conf path does not exist: '$SSMTP_CONF_PATH'"
    exit 1
fi

LEN=${#SERVER} 

if [ $LEN -eq 0 ]
then
	echo "\$SERVER was not defined in tool script: $TOOL_SCRIPT"
    exit 1
fi 
