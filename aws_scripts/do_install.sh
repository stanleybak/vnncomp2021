#!/bin/bash -e
# install tool

source ./tool.sh

pushd ${TOOL_NAME}
./${SCRIPTS_DIR}/install_tool.sh v1
popd
