#!/bin/bash
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
export DNNL_VERBOSE=1
export LD_PRELOAD=`pwd`/libcldump1.so

#export CreateMultipleSubDevices=1
#export ZE_AFFINITY_MASK=0.2
usage() {

echo "exit() : --all --all_ci --fail "
exit
}

if [ "$#" -ne 1 ]; then
usage;
fi




if [ "$1" == "--all" ]; then
./$bn --softmax --engine=gpu --batch=test_softmax_all
exit;
fi

if [ "$1" == "--all_ci" ]; then
#./$bn --eltwise --engine=gpu --batch=test_eltwise_ci
exit;
fi


if [ "$1" == "--fail" ]; then
#./$bn --softmax --engine=gpu --sdt=f32 --ddt=f32 --axis=0 96x1000
./$bn --softmax --engine=gpu --sdt=f64 --ddt=f64 --axis=0 96x1000
./$bn --softmax --engine=gpu --ddt=f64 --axis=0 --inplace=true 96x1000

exit;
fi




echo "ERROR: command not found"
usage;




#export OMP_NUM_THREAD$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2S=1
#./$bn --deconv --engine=gpu --batch=test_deconv_gpu

#./$bn --deconv --engine=gpu --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n
#./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n

#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab 2x2:2x2


