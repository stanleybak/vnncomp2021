#!/bin/bash -e
# download vnncomp and tool

sudo apt-get update
sudo apt-get install -y ssmtp sharutils mutt

if [ ! -d "vnncomp2021" ] 
then
    git clone https://github.com/stanleybak/vnncomp2021
	pushd vnncomp2021
	git checkout 5ae888a59c22841303d4adcb8224318d2da5933b
	#old: f0dcc469c8cf09a00ab5fba57bdbe1e995ae97db
	popd
fi

source ./tool.sh

if [ ! -d ${TOOL_NAME} ]  
then
	if [ ${REPO} != 0 ]
	then
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
		echo "No repo provided... please manually transfer tool code to server (and cadjust permissions of tool scripts if needed)"
	fi
fi
