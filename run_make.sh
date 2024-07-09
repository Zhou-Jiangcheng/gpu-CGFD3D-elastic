#!/bin/bash

#-- SM code of A100
export SMCODE=sm_80

#export CUDAHOME=/data3/lihl/software/cuda-11.5
#export MPIHOME=/data3/lihl/software/openmpi-gnu-4.1.2
#export NETCDF=/data3/lihl/software/disable-netcdf-4.4.1

#-- SM code of A100
export CUDAHOME=/usr/local/cuda-11.8
export MPIHOME=/data/apps/openmpi/4.1.5-cuda-aware
export NETCDF=/data/apps/NetCDF/disable-netcdf-4.8.1

echo
echo "start to make ..."
make -j 
