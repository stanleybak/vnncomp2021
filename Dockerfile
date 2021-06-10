# Dockerfile for vnncomp
# this is an example that uses the tool_example scripts

FROM ubuntu:20.04

RUN echo "Starting..."
RUN apt-get update && apt-get install -y bc git # bc is used in vnncomp measurement scripts

################## install tool

ARG TOOL_NAME=simple_adversarial_generator
ARG SCRIPTS_DIR=vnncomp_scripts
ARG REPO=https://github.com/stanleybak/simple_adversarial_generator.git 
ARG COMMIT=570a28c4d2c35a77e6f9fe6db886cf8381b755b2

RUN git clone $REPO
RUN cd $TOOL_NAME && git checkout $COMMIT && cd ..
RUN /$TOOL_NAME/$SCRIPTS_DIR/install_tool.sh v1

#################### run vnncomp
COPY . /vnncomp2021

# run all categories to produce out.csv
ARG CATEGORIES="test"
RUN vnncomp2021/run_all_categories.sh v1 /$TOOL_NAME/$SCRIPTS_DIR vnncomp2021 out.csv $CATEGORIES
