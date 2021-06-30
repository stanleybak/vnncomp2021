#!/bin/bash
# run code in docker. Single argument: space-seperated list of categories
#
# example usage: ./run_in_docker.sh "acasxu cifar10_resnet cifar2020 eran marabou-cifar10 mnistfc nn4sys oval21 test verivital"

CATEGORIES="test"

if [ "$#" -eq 1 ]; then
    CATEGORIES=$1
fi

PREFIX=vnncomp2021
RESULT_FILE=out.csv

CONTAINER=${PREFIX}_container
IMAGE=${PREFIX}_image

rm -f ${RESULT_FILE}
echo "Running in Docker using container name $CONTAINER and image name $IMAGE, with categories: $CATEGORIES"

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

docker build --build-arg CATEGORIES="$CATEGORIES" . -t $IMAGE

docker run -d --name $CONTAINER $IMAGE tail -f /dev/null

# "docker ps" should now list the image as running

# to get a shell, remove the lines at the end that delete the container and do: "docker exec -it $CONTAINER bash"


docker cp $CONTAINER:/${RESULT_FILE} ${RESULT_FILE}

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

echo "Done. Result file ${RESULT_FILE} should be in the local folder if there were no errors."
