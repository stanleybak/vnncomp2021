#!/bin/bash -e
# setup aws instance
# manually set TOOL_SCRIPT, PEM, and SSMTP_CONF_PATH before starting

if [ "$#" -ne 1 ] 
then
    echo "Expected single arguments (got $#): TOOL_NAME"
    exit 1
fi

export TOOL=$1
source step0_config.sh



ssh -i $PEM ${SERVER} 'mkdir -p ~/work'

scp -i ${PEM} $TOOL_SCRIPT ${SERVER}:~/work/tool.sh
scp -i ${PEM} do_download.sh ${SERVER}:~/work/
scp -i ${PEM} do_install.sh ${SERVER}:~/work/
scp -i ${PEM} schedule_shutdown.sh ${SERVER}:~/work/
scp -i ${PEM} run_all.sh ${SERVER}:~/work/

ssh -i $PEM ${SERVER} 'cd work;./do_download.sh'

# copy ssmtp setup for email alerts (must be AFTER running do_download.sh which installs ssmtp)
cat $SSMTP_CONF_PATH | ssh -i $PEM ${SERVER} "sudo tee /etc/ssmtp/ssmtp.conf > /dev/null"

if [[ ${REPO} == 0 ]]
then
    printf "\nCommand to copy tool: scp -i ${PEM} filename.tar.gz ${SERVER}:~/work/\n"
fi

echo "Copying and generic setup done. To continue, check tool's intallation instructions and do: ./step2_install.sh $TOOL"
