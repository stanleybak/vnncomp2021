#!/bin/bash
# run measurements for all categories for a single tool (passed on command line)
# four args: 'v1' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file
#
# for example ./run_all_categories.sh v1 ../tool_example .. ./out_test.csv

# list of benchmark category names seperated by spaces
CATEGORY_LIST="test acasxu"
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

TOOL_FOLDER=$2
VNNCOMP_FOLDER=$3
RESULT_CSV_FILE=$4

if [[ $RESULT_CSV_FILE != *csv ]]; then
	echo "result csv file '$RESULT_CSV_FILE' should end in .csv"
	exit 1
fi

if [ ! -d $VNNCOMP_FOLDER ] 
then
    echo "VNNCOMP directory does not exist: '$VNNCOMP_FOLDER'" 
    exit 1
fi

if [ ! -d $TOOL_FOLDER ] 
then
    echo "Tool scripts directory does not exist: '$TOOL_FOLDER'" 
    exit 1
fi

echo "Running measurements with vnncomp folder '$VNNCOMP_FOLDER' for tool scripts in '$TOOL_FOLDER'. Saving results to '$RESULT_CSV_FILE'."

# run on each benchmark category
for CATEGORY in $CATEGORY_LIST
do
    INSTANCES_CSV_PATH="${VNNCOMP_FOLDER}/benchmarks/${CATEGORY}/${CATEGORY}_instances.csv"
    echo "Running $CATEGORY category from $INSTANCES_CSV_PATH"
    
    # loop through csv file and run on each instance in category
    PREV_IFS=$IFS
    IFS=','
    if [ ! -f $INSTANCES_CSV_PATH ]
    then
	    echo "$INSTANCES_CSV_PATH file not found"
	    exit 1
    fi
	    
    while read ONNX VNNLIB TIMEOUT
    do
	echo "onnx : $ONNX"
	echo "vnnlib : $VNNLIB"
	echo "timeout : $TIMEOUT"
	run_single_instance.sh v1 $TOOL_FOLDER $CATEGORY $ONNX $VNNLIB $TIMEOUT $RESULT_CSV_FILE
		
	done < $INSTANCES_CSV_PATH
	IFS=$PREV_IFS
done
