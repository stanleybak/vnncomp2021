
#!/bin/bash

# You can get a free key from https://www.gurobi.com/downloads/free-academic-license/
# It looks like: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

SERVER=$1
GROUP=$2
USERNAME=$3
PASSWARD=$4
GUROBI_USER=$5
KEY=$6

OPENCONNECT_PID=""

function startOpenConnect(){
    # start here open connect with your params and grab its pid
    echo "${PASSWARD}" | openconnect -b -q "${SERVER}" --authgroup "${GROUP}" -u "${USERNAME}" --passwd-on-stdin & OPENCONNECT_PID=$!
}

function checkOpenconnect(){
    ps -p "${OPENCONNECT_PID}"
    # print the status so we can check in the main loop
    echo $?
}

startOpenConnect
sleep 2
OPENCONNECT_STATUS=$(checkOpenconnect)
echo $OPENCONNECT_STATUS
echo | sudo -u $GUROBI_USER /opt/gurobi912/linux64/bin/grbgetkey $KEY
sudo killall -SIGINT openconnect

