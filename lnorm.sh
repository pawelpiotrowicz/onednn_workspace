#!/bin/bash
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
#export DNNL_VERBOSE=1
#export DNNL_VERBOSE=debuginfo=10
#export LD_PRELOAD=`pwd`/libcldump1.so

#export CreateMultipleSubDevices=1
#export ZE_AFFINITY_MASK=0.2


prim="lnorm"


usage() {

echo "exit() : --all --ci --gpu --fail "
exit
}

if [ "$#" -ne 1 ]; then
usage;
fi


if [ "$1" == "--gdb" ]; then

cgdb --args ./$bn --${prim} --dir=BWD_D --sdt=f64 --axis=0 16x16_nbnorm_ci_0d:0
	exit;
fi

if [ "$1" == "--all" ]; then
./$bn --${prim} --engine=gpu --batch=test_${prim}_all
exit;
fi

if [ "$1" == "--gpu" ]; then
./$bn --${prim} --engine=gpu --batch=test_${prim}_gpu
exit;
fi

if [ "$1" == "--ci" ]; then
./$bn --${prim} --engine=gpu --batch=test_${prim}_ci
exit;
fi


failed() {

./$bn --${prim} --engine=gpu --sdt=f64 --ddt=f64 96x1000
./$bn --${prim} --engine=gpu  96x1000
./$bn --${prim} --engine=gpu --ddt=f64 --alg=LOGSOFTMAX --inplace=true 96x1000
./$bn --${prim} --engine=gpu --sdt=f64 --ddt=f64 --alg=LOGSOFTMAX 96x1000
./$bn --${prim} --engine=gpu --ddt=f64 --alg=LOGSOFTMAX 96x1000
./$bn  --${prim} --engine=gpu --ddt=f64 6144x48
./$bn --${prim} --engine=gpu --dir=BWD_D --sdt=bf16 --ddt=bf16 --alg=LOGSOFTMAX --axis=3 2x16x128x128
}

if [ "$1" == "--fail" ]; then
#./$bn --${prim} --engine=gpu --dt=f16 --tag=nhwc --flags=CH mb4ic160ih160
#./$bn --${prim} --engine=gpu --dir=BWD_DW --dt=f16 ic256ih56
./$bn --${prim} --engine=gpu --dir=BWD_DW --dt=f16:f16 --tag=abc 1x8192x1024
./$bn --${prim} --engine=gpu --skip-impl=ref --dir=BWD_DW --dt=f16:f16 --tag=abc 1x8192x1024
#./$bn --ip --engine=gpu --dir=BWD_WB --cfg=f16 --stag=ab --wtag=ba --dtag=ab mb11456ic96oc16 
#./$bn --ip --engine=gpu --dir=BWD_WB --cfg=f32 --stag=ab --wtag=ba --dtag=ab mb11456ic96oc16 
#./$bn --${prim} --engine=gpu --sdt=f32 --ddt=f32 --axis=0 96x1000
#./$bn --${prim} --engine=gpu --sdt=f64 --ddt=f64 --axis=0 96x1000
#./$bn --${prim} --engine=gpu --ddt=f64 --axis=0 --inplace=true 96x1000
#./$bn  --${prim} --engine=gpu --skip-impl=ref --ddt=f64 96x1000
#./$bn --${prim} --engine=gpu --dir=BWD_D --axis=0 --inplace=true 8192x64
#./$bn --${prim} --engine=gpu --sdt=bf16 --ddt=f64 --alg=LOGSOFTMAX 16x16_n"bnorm_ci_0d:0"
#./$bn --${prim} --engine=gpu --sdt=f16 --ddt=f64 16x16_n"bnorm_ci_0d:0"
#./$bn --${prim} --engine=gpu --dir=BWD_D --sdt=f64 --ddt=f64 --alg=LOGSOFTMAX --axis=3 --inplace=true 2x16x128x2x4
#./$bn --${prim} --engine=gpu --sdt=f64 --ddt=f64 --alg=LOGSOFTMAX --axis=3 --inplace=true 2x16x128x2x4
#./$bn --${prim} --engine=gpu --sdt=bf16 --ddt=f64 --alg=LOGSOFTMAX --axis=3 --inplace=true 2x16x128x2x4
#./$bn --${prim} --engine=gpu --dir=BWD_D --sdt=bf16 --ddt=f64 --alg=LOGSOFTMAX --axis=3 --inplace=true 2x16x128x2x4
#./$bn --${prim} --engine=gpu --ddt=f64 --axis=0 16x16_n"bnorm_ci_0d:0" 
#failed;

#cgdb --args ./$bn --${prim} --engine=gpu --ddt=f64 --alg=LOGSOFTMAX --axis=0 --inplace=true 96x1000
#cgdb --args ./$bn --${prim} --engine=gpu  --alg=LOGSOFTMAX --axis=0 --inplace=true 96x1000
#./$bn --${prim} --engine=gpu --ddt=f64 --alg=LOGSOFTMAX --axis=0 --inplace=true 96x1000
#./$bn --${prim} --engine=gpu --ddt=f64 --alg=LOGSOFTMAX --axis=0 96x1000
#./$bn --${prim} --engine=gpu  --alg=LOGSOFTMAX --axis=0 --inplace=true 96x1000
#./$bn  --${prim} --engine=gpu --ddt=f64 6144x48
#./$bn --${prim} --engine=gpu --dir=BWD_D --sdt=bf16 --ddt=bf16 --alg=LOGSOFTMAX --axis=3 2x16x128x128


#./$bn --${prim} --engine=gpu --dir=BWD_D --sdt=f64 --axis=0 16x16_nbnorm_ci_0d:0
#./$bn --${prim}  --sdt=f64 --axis=0 16x16_nbnorm_ci_0d:0


exit;
fi




echo "ERROR: command not found"
usage;




#export OMP_NUM_THREAD$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2S=1
#./$bn --deconv --engine=gpu --batch=test_deconv_gpu

#./$bn --deconv --engine=gpu --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n
#./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n

#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab 2x2:2x2


