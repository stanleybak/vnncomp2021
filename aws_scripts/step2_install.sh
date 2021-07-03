#!/bin/bash -e
# setup aws instance
# before starting, make sure you can connect with ssh (and it accepts fingerprint)

source step0_config.sh

ssh -i $PEM ${SERVER} 'cd work;./do_install.sh 2>&1' | tee install_${TOOL}_log.txt

echo "tool setup done. Complete any manual steps before step 3 (test)."
