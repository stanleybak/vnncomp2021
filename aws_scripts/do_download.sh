#!/bin/bash -e
# download vnncomp and tool

sudo apt-get update
sudo apt-get install -y ssmtp sharutils mutt

sudo rm -frv vnncomp2021

VNNCOMP_REPO=https://github.com/stanleybak/vnncomp2021
VNNCOMP_COMMIT=9ad96ea25fa9b5ea2ce2e56943f17267dab91e49

echo "Cloning vnncomp repo ${VNNCOMP_REPO} with commit hash ${VNNCOMP_COMMIT}"

git clone $VNNCOMP_REPO
pushd vnncomp2021
git checkout $VNNCOMP_COMMIT
popd

source ./tool.sh

sudo rm -frv $TOOL_NAME

if [ ${REPO} != 0 ]
then
	echo "Cloning ${TOOL_NAME} from ${REPO} with commit hash ${COMMIT}"

	git clone ${REPO} ${TOOL_NAME}
	pushd ${TOOL_NAME}
	git checkout ${COMMIT}
	
	############
	if [ ! -f "./$SCRIPTS_DIR/install_tool.sh" ] 
	then
		echo "tool script does not exist: ./$SCRIPTS_DIR/install_tool.sh"
		exit 1
	else
		chmod +x "./$SCRIPTS_DIR/install_tool.sh"
	fi
	
	#############
	if [ ! -f "./$SCRIPTS_DIR/prepare_instance.sh" ] 
	then
		echo "tool script does not exist: ./$SCRIPTS_DIR/prepare_instance.sh"
		exit 1
	else
		chmod +x "./$SCRIPTS_DIR/prepare_instance.sh"
	fi
	
	#############
	if [ ! -f "./$SCRIPTS_DIR/run_instance.sh" ] 
	then
		echo "tool script does not exist: ./$SCRIPTS_DIR/run_instance.sh"
		exit 1
	else
		chmod +x "./$SCRIPTS_DIR/run_instance.sh"
	fi
	
	popd
else
	echo "No repo provided... please manually transfer tool code to server (and adjust permissions of tool scripts if needed)"
fi
