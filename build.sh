#!/bin/bash


nrepo="oneDNN"
dbuild="build"

if [ ! -d "$nrepo" ]; then
  echo "Direcotry exists $nrepo"
   git clone https://github.com/oneapi-src/oneDNN.git $nrepo
fi



cd $nrepo


if [ ! -d "$dbuild" ]; then
   mkdir $dbuil
fi

cd $dbuild

cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-DDEBUG_PRINT=1"  -DDNNL_GPU_RUNTIME=OCL -DDNNL_CPU_RUNTIME=OMP ..

make -j12






