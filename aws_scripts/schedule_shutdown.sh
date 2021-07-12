#!/bin/bash
# shutdown after X seconds

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    echo "Expected 3-4 arguments (got $#): secs, email_subject, file1, file2"
    exit 1
fi

SECS=$1
SUBJECT=$2
FILE1=$3
FILE2=$4

sleep $SECS
echo "$SUBJECT" | mutt -s $SUBJECT -a $FILE1 $FILE2 -- stanleybak@gmail.com
sleep 2
sudo shutdown -h now 
