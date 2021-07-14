#!/bin/bash -e
# copy complete stdout.txt from server to local
# in theory, first 10 mb will be emailed as well during stpe 4, so this may not be necessary


if [ "$#" -ne 1 ] 
then
    echo "Expected single argument (got $#): TOOL_NAME"
    exit 1
fi

export TOOL=$1
source step0_config.sh

DEST="stdout_${TOOL}.txt"

scp -i ${PEM} ${SERVER}:~/work/vnncomp2021/stdout.txt $DEST

echo "Copied stdout for $TOOL to ${DEST}. Remember to stop the instance."
