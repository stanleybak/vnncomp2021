#!/bin/bash -ve
# setup aws instance
# before starting, make sure you can connect with ssh (and it accepts fingerprint)

source step0_config.sh

ssh -i $PEM ${SERVER} 'cd work;./do_install.sh'

echo "tool setup done. Complete any manual steps before step 3 (test)."
