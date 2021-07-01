#!/bin/bash -ve
# download vnncomp and tool

sudo apt-get update
sudo apt-get install -y ssmtp sharutils mutt

if [ ! -d "vnncomp2021" ] 
then
    git clone https://github.com/stanleybak/vnncomp2021
	pushd vnncomp2021
	git checkout 27c3cc7cdce313cfeb10a6b44c37bde82509f21f
	popd
fi

source ./tool.sh

if [ ! -d ${TOOL_NAME} ] 
then
	git clone ${REPO} ${TOOL_NAME}
	pushd ${TOOL_NAME}
	git checkout ${COMMIT}
	popd
fi
