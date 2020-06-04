#!/bin/bash
#
# This is a installation file prepared to install all dependencies
# for the lrp-enabled caffe code as part of the LRP Toolbox v1.0 on
# systems running the wide spread Ubuntu 14.04 LTS 64 bit distribution.
# Since Ubuntu is the standard release platform for Caffe, this script
# might be able to install the lrp-enabled code as well on different 
# releases. There is, however, no guarantee for that. This is especially
# true for other Linux derivates or OSs in general. This script should
# therefore only be considered a convenience solution for installing
# the lrp-enabled caffe code on the previously mentioned OS
#
# Before executing this script, please read and modify the following commands 
# carefully in order to prevent unwanted changes to your system.
#
# This installation requires administrator level privileges.
# the following routine ist based on the information available at:
#
# http://caffe.berkeleyvision.org/install_apt.html



# INSTALL GENERAL DEPENDENCIES
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler build-essential
sudo apt-get install -y --no-install-recommends libboost-all-dev



# INSTALL CUDA
#
# fetch and install cuda 7.5 locally instead of using apt-get
# the following wget command is equivalent to choosing the appropriate download
# (debian package, local) for Ubuntu 14.04 LTS from
#
# https://developer.nvidia.com/cuda-downloads
#
# we have opted by default for the larger debian package due to the non-requirement
# of user interaction during the installation process.
#
# if the runfile installation or a system-wide package installation is preferred,
# comment/uncomment the following lines accodringly.
# cuda headers are required for building caffe.

echo "Cuda installation disabled. This is something you better do yourself, after all."

# prepare download folder
#mkdir cuda_dl ; cd cuda_dl


# local debian package installation. caution, causes an update of installed packages! 
#wget -nc http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
#dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
#apt-get update
#apt-get install cuda

# network-based debian package installation. disabled by default.
#wget -nc http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
#dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
#apt-get update
#apt-get install cuda

# local installation using a runfile
#wget -nc http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
#sh cuda_7.5.18_linux.run

# cleanup after cuda installation
#cd .. ; rm -r cuda_dl



# INSTALL ATLAS AND REMAINING DEPENDENCIES
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

#the code as originally been written for and on ubuntu 14.04
#this script here modifies includes and library names for caffe and the demonstator to compile on 16.04. will do nothing on 14.04 and only execute once.
bash caffe-lrp-16.04-patch.sh


# COMPILE CAFFE CODE
make clean
make all -j10
#test your build?
#make test
#make runtest


# BUILD DEMONSTRATOR APPLICATION (requires and installs ImageMagick)
cd demonstrator
sudo apt-get install -y libmagick++-dev
bash build.sh
chmod +x lrp_demo



# DOWNLOAD BVLC REFERENCE CAFFE MODEL AND START DEMONSTRATOR
bash download_model.sh
./lrp_demo ./config_sequential.txt ./testfilelist.txt ./

echo "output images can be found in $(pwd)/lrp_output"
cd ..

# MAKE PYCAFFE AND SHOW DEMO USAGE
# notes:
#   - the pycaffe interface only supports python 2, you might have to adapt the python command
#   - only temporarily adds the caffe-master-lrp/python directory to your PYTHONPATH. If you want to use the lrp caffe python wrapper from outside this script, add it to your PYTHONPATH manually.

make pycaffe
cd demonstrator
python2 lrp_python_demo.py
