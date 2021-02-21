#!/bin/bash
# example install_tool.sh script for VNNCOMP for simple_adversarial_generator (https://github.com/stanleybak/simple_adversarial_generator) 
# Stanley Bak, Feb 2021

TOOL_NAME=simple_adv_gen
REPO_URL=https://github.com/stanleybak/simple_adversarial_generator 
COMMIT=4ee8d608192747f3292420383d1d170fa93d51a1
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

# install the tool if it doesn't exist
if [ ! -d $TOOL_NAME ] 
then
	# get the code and install dependencies
	git clone ${REPO_URL} ${TOOL_NAME} && 
	cd ${TOOL_NAME} && 
	git checkout ${COMMIT} &&
	pip3 install -r requirements.txt
fi
