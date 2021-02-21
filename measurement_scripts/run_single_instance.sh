#!/bin/bash
# run single instance for a single tool and accumulate result to an output csv file
#
# args: 'v1' (version string), tool_scripts_folder, category, onnx_file, vnnlib_file, timeout, result_csv_file

VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

TOOL_FOLDER=$2
CATEGORY=$3
ONNX=$4
VNNLIB=$5
TIMEOUT=$6
RESULT_CSV_FILE=$7

if [[ $RESULT_CSV_FILE != *csv ]]; then
	echo "result csv file '$RESULT_CSV_FILE' should end in .csv"
	exit 1
fi

echo "Running '$CATEGORY' measurement on network '$ONNX' with vnnlin file '$VNNLIB' and timeout '$TIMEOUT'"

${TOOL_FOLDER}/prepare_instance.sh "v1" "$CATEGORY" "$ONNX" "$VNNLIB"
EXIT_CODE=$?
echo "Prepare Exit Code: $EXIT_CODE"

# run on benchmarks
START=$(date +%s.%N)
${TOOL_FOLDER}/run_instance.sh "v1" "$CATEGORY" "$ONNX" "$VNNLIB" "out.txt" "$TIMEOUT"
EXIT_CODE=$?
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

RESULT=$(head -n 1 "out.txt")

# remove whitespace
RESULT=${RESULT//[[:space:]]/}

echo "Exit Code: $EXIT_CODE"
echo "Resuts: $RESULT"
echo "Runtime: $DIFF"

