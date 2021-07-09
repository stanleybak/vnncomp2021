#!/bin/bash -e
# setup aws instance
# before starting, make sure you can connect with ssh (and it accepts fingerprint)

if [ "$#" -ne 1 ] 
then
    echo "Expected single arguments (got $#): TOOL_NAME"
    exit 1
fi

export TOOL=$1
source step0_config.sh

ssh -i $PEM ${SERVER} 'cd work;./do_install.sh 2>&1' | tee install_${TOOL}_log.txt

echo "tool setup done. Complete any manual steps and then do: ./step3_test.sh $TOOL"
