# Dockerfile for vnncomp
# this is an example that uses the tool_example scripts

FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3 python3-pip

RUN git clone https://github.com/stanleybak/vnncomp2021.git

# for testing different tools, you can add git clone here or wget or otherwise get your tool's files and scripts

# setup tool using install_tool.sh
RUN vnncomp2021/tool_example/install_tool.sh v1

RUN git pull

# run all categories to produce out.csv
RUN vnncomp2021/measurement_scripts/run_all_categories.sh v1 vnncomp2021/tool_example vnncomp2021 out.csv
