#!/bin/bash


nrepo="libraries.performance.math.onednn"
dbuild="build"

usage() {
 echo "usage : --debug --jcores=10"
 exit
}

OPT_CORES=20
OPT_DEBUG=""
OPT_VERBOSE=0
OPT_CLEAR=0
for i in "$@"; do
  case $i in
    -j=*|--jcores=*)
      OPT_CORES="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--searchpath=*)
      SEARCHPATH="${i#*=}"
      shift # past argument=value
      ;;
    -l=*|--lib=*)
      LIBPATH="${i#*=}"
      shift # past argument=value
      ;;
    --default)
      DEFAULT=YES
      shift # past argument with no value
      ;;
    --verbose)
      OPT_VERBOSE=1
      shift
      ;;
      --help)
       usage;
       shift
      ;;
    --skipcmake)
      OPT_SKIPCMAKE=1
      shift
      ;;
    --clear)
      OPT_CLEAR=1
      shift
      ;;
    --debug)
      OPT_DEBUG="-DCMAKE_BUILD_TYPE=Debug"
      shift
      ;;
    --release)
      OPT_RELEASE=1
      ;;
    *)
      # unknown option
      echo "Unknown options $i"
      exit
      ;;
  esac
done
















if [ ! -d "$nrepo" ]; then
  echo "Direcotry exists $nrepo"
   git clone https://github.com/oneapi-src/oneDNN.git $nrepo
fi



cd $nrepo


if [ ! -d "$dbuild" ]; then
   mkdir $dbuild
fi

cd $dbuild

cmd="cmake $OPT_DEBUG -DCMAKE_CXX_FLAGS=\"-DDEBUG_PRINT=1\"  -DDNNL_GPU_RUNTIME=OCL -DDNNL_CPU_RUNTIME=OMP .."

echo "$cmd"

$cmd

make -j$OPT_CORES






