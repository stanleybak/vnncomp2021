#!/bin/bash -e
# setup aws instance
# before starting, make sure you can connect with ssh (and it accepts fingerprint)

source step0_config.sh

if [ ! -f $TOOL_SCRIPT ] 
then
    echo "Tool script does not exist: '$TOOL_SCRIPT'"
    exit 1
fi

if [ ! -f $PEM ] 
then
    echo "PEM path does not exist: '$PEM'"
    exit 1
fi

if [ ! -f $SSMTP_CONF_PATH ] 
then
    echo "SSMTP conf path does not exist: '$SSMTP_CONF_PATH'"
    exit 1
fi

ssh -i $PEM ${SERVER} 'mkdir -p ~/work'

scp -i ${PEM} $TOOL_SCRIPT ${SERVER}:~/work/tool.sh
scp -i ${PEM} do_download.sh ${SERVER}:~/work/
scp -i ${PEM} do_install.sh ${SERVER}:~/work/
scp -i ${PEM} schedule_shutdown.sh ${SERVER}:~/work/
scp -i ${PEM} run_all.sh ${SERVER}:~/work/

ssh -i $PEM ${SERVER} 'cd work;./do_download.sh'

# copy ssmtp setup for email alerts (must be AFTER running do_download.sh which installs ssmtp)
cat $SSMTP_CONF_PATH | ssh -i $PEM ${SERVER} "sudo tee /etc/ssmtp/ssmtp.conf > /dev/null"


echo "generic setup done. proceed with tool setup at step 2"
