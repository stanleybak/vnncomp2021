#!/bin/bash -ve
# setup aws instance
# before starting, make sure you can connect with ssh (and it accepts fingerprint)

source step0_config.sh

TOOL_SCRIPT=/home/stan/repositories/vnncomp2021/tools/${TOOL}.sh

if [ ! -f $TOOL_SCRIPT ] 
then
    echo "Tool script does not exist: '$TOOL_SCRIPT'"
    exit 1
fi

ssh -i $PEM ${SERVER} 'mkdir -p ~/work'

scp -i ${PEM} $TOOL_SCRIPT ${SERVER}:~/work/tool.sh
scp -i ${PEM} do_download.sh ${SERVER}:~/work/
scp -i ${PEM} do_install.sh ${SERVER}:~/work/
scp -i ${PEM} schedule_shutdown.sh ${SERVER}:~/work/
scp -i ${PEM} run_all.sh ${SERVER}:~/work/

# copy ssmtp setup for email alerts
sudo cat /etc/ssmtp/ssmtp.conf | ssh -i $PEM ${SERVER} "sudo tee /etc/ssmtp/ssmtp.conf > /dev/null"

ssh -i $PEM ${SERVER} 'cd work;./do_download.sh'

echo "generic setup done. proceed with tool setup at step 2"
