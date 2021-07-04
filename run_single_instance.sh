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

echo "Doing run_single_instance with category '$CATEGORY' on onnx network '$ONNX' with vnnlib file '$VNNLIB' and timeout '$TIMEOUT'"
RESULT_STR="run_single_instance_error"
RUNTIME=-1

# uncompress onnx file if needed: "gzip -dk test_zipped.onnx.gz"
if [[ $ONNX == *gz ]] # * is used for pattern matching
then
	UNCOMPRESSED_ONNX=${ONNX%.gz}
	
	if [ ! -f $UNCOMPRESSED_ONNX ]
    then
	    # UNCOMPRESSED_ONNX doesn't exist, create it
	    echo "$UNCOMPRESSED_ONNX doesn't exist, unzipping"
	    gzip -vdk $ONNX
    fi
    
    ONNX=$UNCOMPRESSED_ONNX
fi

# timeout upon which we'll kill the process
KILL_TIMEOUT=$(echo "$TIMEOUT + 60" | bc)

# run prepare instance (60 second timeout)
PREPARE_INSTANCE_TIMEOUT=60

START=$(date +%s.%N)
timeout $PREPARE_INSTANCE_TIMEOUT ${TOOL_FOLDER}/prepare_instance.sh "v1" "$CATEGORY" "$ONNX" "$VNNLIB"
EXIT_CODE=$?
END=$(date +%s.%N)
PREPARE_RUNTIME=$(echo "$END - $START" | bc)
	
echo "prepare_instance.sh exit code: $EXIT_CODE, runtime: $PREPARE_RUNTIME"

if [ 0 != ${EXIT_CODE} ]; then
	RESULT_STR="prepare_instance_error_$EXIT_CODE"
	
	if [ 124 == ${EXIT_CODE} ]; then
	    echo "Error: prepare_instance.sh exceeded $PREPARE_INSTANCE_TIMEOUT second timeout!"
	    RESULT_STR="prepare_instance_timeout"
	fi
else
	echo "no_result_in_file" > "out.txt"
	# run on benchmarks
	START=$(date +%s.%N)
	timeout $KILL_TIMEOUT ${TOOL_FOLDER}/run_instance.sh "v1" "$CATEGORY" "$ONNX" "$VNNLIB" "out.txt" "$TIMEOUT"
	
	EXIT_CODE=$?
	END=$(date +%s.%N)
	RUNTIME=$(echo "$END - $START" | bc)
	
	if [ 0 != ${EXIT_CODE} ]; then
		RESULT_STR="error_exit_code_${EXIT_CODE}"
		
		if [ 124 == ${EXIT_CODE} ]; then
		    echo "Error: run_instance.sh exceeded $KILL_TIMEOUT second timeout!"
		    RESULT_STR="run_instance_timeout"
		fi
	else
		RESULT=$(head -n 1 "out.txt")

		# remove whitespace
		RESULT_STR=${RESULT//[[:space:]]/}
	fi

	echo "run_instance.sh exit code: ${EXIT_CODE}, Result: ${RESULT_STR}, Runtime: ${RUNTIME}"
fi

echo "Appending result '$RESULT_STR' to csv file '$RESULT_CSV_FILE'"
echo "${CATEGORY},${ONNX},${VNNLIB},${PREPARE_RUNTIME},${RESULT_STR},${RUNTIME}" >> $RESULT_CSV_FILE


