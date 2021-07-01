#!/bin/bash
# run measurements (should be run in tmux)

source tool.sh

pushd vnncomp2021

./run_all_categories.sh v1 ~/work/${TOOL_NAME}/$SCRIPTS_DIR . ./out.csv "acasxu cifar10_resnet cifar2020 eran marabou-cifar10 mnistfc nn4sys oval21 test verivital" | tee stdout.txt

popd
./schedule_shutdown.sh 1 "Finished_$TOOL_NAME" vnncomp2021/out.csv
