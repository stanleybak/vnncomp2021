#!/bin/bash
# run tool measurements on aws instance
# total timeout: 240000 secs (~66 hours)

source step0_config.sh

ssh -i $PEM ${SERVER} "cd ~/work/vnncomp2021; tmux new -d -s auto_shutdown ../schedule_shutdown.sh 240000 'Timeout_${TOOL}' out.csv"

ssh -i $PEM ${SERVER} 'cd ~/work; tmux new -d -s measurements ./run_all.sh'

echo "Measurements started for $TOOL. Results should come over email and instance should shut down automatically."

echo "You can track progress by conneting to the instance and connecting to the tmux session (ctrl+b, d to dettach):"
echo "ssh -i \"${PEM}\" ${SERVER} tmux attach-session -t measurements"
