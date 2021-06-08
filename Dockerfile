# Dockerfile for vnncomp
# this is an example that uses the tool_example scripts

FROM ubuntu:20.04

RUN echo "Starting..."
RUN apt-get update
RUN apt-get install -y bc # bc is used in vnncomp measurement scripts

################## install tool
RUN apt-get install -y git

ARG TOOL_NAME=simple_adversarial_generator
ARG SCRIPTS_DIR=vnncomp_scripts
ARG REPO=https://github.com/stanleybak/simple_adversarial_generator.git 
ARG COMMIT=83bf04757eb52608696705337f16c58b0aa97995
ARG CATEGORIES="test acasxu"

RUN git clone $REPO
RUN cd $TOOL_NAME && git checkout $COMMIT && cd ..
RUN /$TOOL_NAME/$SCRIPTS_DIR/install_tool.sh v1

#################### run vnncomp
COPY . /vnncomp2021

# run all categories to produce out.csv
RUN vnncomp2021/run_all_categories.sh v1 /$TOOL_NAME/$SCRIPTS_DIR vnncomp2021 out.csv $CATEGORIES
