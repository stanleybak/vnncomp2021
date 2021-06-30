#!/bin/bash
# run measurements for all categories for a single tool (passed on command line)
# four args: 'v1' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file
#
# for example ./run_all_categories.sh v1 ~/repositories/simple_adversarial_generator/vnncomp_scripts . ./out.csv "test acasxu"

VERSION_STRING=v1
SCRIPT_PATH=$(dirname $(realpath $0))

MAX_CATEGORY_TIMEOUT=6*60*60
MIN_CATEGORY_TIMEOUT=3*60*60

# if this is "true", will only report total timeout (and not run anything)
TOTAL_TIMEOUT_ONLY="false"
TOTAL_TIMEOUT=0

# if "true", measure overhead after each category
MEASURE_OVERHEAD="true"

# check arguments
if [ "$#" -ne 5 ] && [ "$#" -ne 6 ]; then
    echo "Expected 5-6 arguments (got $#): '$VERSION_STRING' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file, categories, single_run (0/1, if 1 then run the first instance of each category only)"
    exit 1
fi

if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

TOOL_FOLDER=$2
VNNCOMP_FOLDER=$3
RESULT_CSV_FILE=$4
# list of benchmark category names seperated by spaces
CATEGORY_LIST=$5
SINGLE_RUN=$6

# if "true", only run the first benchmark instance in each category (for testing)
FIRST_INSTANCE_ONLY="false"

if [[ $SINGLE_RUN == "1" ]]; then
    FIRST_INSTANCE_ONLY="true"
fi

if [[ $RESULT_CSV_FILE != *csv ]]; then
    echo "result csv file '$RESULT_CSV_FILE' should end in .csv"
    exit 1
fi

if [ ! -d $VNNCOMP_FOLDER ] 
then
    echo "VNNCOMP directory does not exist: '$VNNCOMP_FOLDER'" 
    echo "errored" > $RESULT_CSV_FILE
    exit 0
fi

if [ ! -d $TOOL_FOLDER ] 
then
    echo "Tool scripts directory does not exist: '$TOOL_FOLDER'" 
    echo "errored" > $RESULT_CSV_FILE
    exit 0
fi

echo "Running measurements with vnncomp folder '$VNNCOMP_FOLDER' for tool scripts in '$TOOL_FOLDER' and saving results to '$RESULT_CSV_FILE'."

# clear file
echo -n "" > $RESULT_CSV_FILE

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
	    
	    echo "errored" > $RESULT_CSV_FILE
	    exit 0
    fi
    
    SUM_TIMEOUT=`awk -F"," '{x+=$3}END{print x}' < $INSTANCES_CSV_PATH`
    echo "Category '$CATEGORY' timeout sum: $SUM_TIMEOUT seconds"
    TOTAL_TIMEOUT=$(( $TOTAL_TIMEOUT + $SUM_TIMEOUT ))
   
    if (( $(echo "$SUM_TIMEOUT < $MIN_CATEGORY_TIMEOUT || $SUM_TIMEOUT > $MAX_CATEGORY_TIMEOUT" |bc -l) )); then
    
	# to compare more closely with last year, ignore runtime threshold for this one
	if [[ $CATEGORY != "cifar2020" && $CATEGORY != "test" ]]; then
	    echo "$CATEGORY sum timeout ($SUM_TIMEOUT) not in valid range [$MIN_CATEGORY_TIMEOUT, $MAX_CATEGORY_TIMEOUT]"
	    
	    echo "errored" > $RESULT_CSV_FILE
	    exit 0
	else
	    echo "Ignoring out of bounds timeout for category $CATEGORY"
	fi
    fi
    
    if [[ $TOTAL_TIMEOUT_ONLY == "true" ]]; then
	continue
    fi
	
    while read ONNX VNNLIB TIMEOUT_CR
    do
	ONNX_PATH="${VNNCOMP_FOLDER}/benchmarks/${CATEGORY}/${ONNX}"
	VNNLIB_PATH="${VNNCOMP_FOLDER}/benchmarks/${CATEGORY}/${VNNLIB}"
	
	# remove carriage return from timeout
	TIMEOUT=$(echo $TIMEOUT_CR | sed -e 's/\r//g')
	
	$SCRIPT_PATH/run_single_instance.sh v1 $TOOL_FOLDER $CATEGORY $ONNX_PATH $VNNLIB_PATH $TIMEOUT $RESULT_CSV_FILE
	
	if [[ $FIRST_INSTANCE_ONLY == "true" && $CATEGORY != "test" ]]; then
	    break
	fi
		
    done < $INSTANCES_CSV_PATH
    IFS=$PREV_IFS
	
    if [[ $MEASURE_OVERHEAD == "true" && $FIRST_INSTANCE_ONLY != "true" ]]; then
	# measure overhead at end (hardcoded model)
	ONNX_PATH="${VNNCOMP_FOLDER}/benchmarks/test/test_tiny.onnx"
	VNNLIB_PATH="${VNNCOMP_FOLDER}/benchmarks/test/test_tiny.vnnlib"
	TIMEOUT=120
	$SCRIPT_PATH/run_single_instance.sh v1 $TOOL_FOLDER $CATEGORY $ONNX_PATH $VNNLIB_PATH $TIMEOUT $RESULT_CSV_FILE
    fi

done

if [[ $TOTAL_TIMEOUT_ONLY == "true" ]]; then
    echo "Total Timeout of all benchmarks: $TOTAL_TIMEOUT"
fi
