#!/bin/bash -ve
# install tool

source ./tool.sh

pushd ${TOOL_NAME}
sudo ./${SCRIPTS_DIR}/install_tool.sh v1
popd
