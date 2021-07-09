#!/bin/bash
# test tool on aws instance

if [ "$#" -ne 1 ] 
then
    echo "Expected single arguments (got $#): TOOL_NAME"
    exit 1
fi

export TOOL=$1
source step0_config.sh

ssh -i $PEM ${SERVER} 'cd ~/work;source tool.sh;cd vnncomp2021;./run_all_categories.sh v1 ~/work/${TOOL_NAME}/$SCRIPTS_DIR . ./out_test.csv "test" 2>&1; cat out_test.csv' | tee test_${TOOL}_log.txt

echo "test done. If content with results, proceed to step 4 by doing: ./step4_run.sh $TOOL"
