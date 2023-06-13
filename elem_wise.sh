#!/bin/bash
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
#export DNNL_VERBOSE=1
#export LD_PRELOAD=`pwd`/libcldump1.so

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
./$bn --eltwise --engine=gpu --batch=test_eltwise_all
exit;
fi

if [ "$1" == "--all_ci" ]; then
./$bn --eltwise --engine=gpu --batch=test_eltwise_ci
exit;
fi


if [ "$1" == "--fail" ]; then
./$bn --eltwise --engine=gpu --dir=BWD_D --dt=f64 --alg=abs --alpha=0 --beta=0 16x16x2x1
./$bn --eltwise --engine=gpu  --skip-impl=ref --dt=f32 --alg=log --alpha=0 --beta=0 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=FWD_I --dt=s32 --tag=axb --alg=relu --alpha=-0.25 --beta=0 3x17x2x5x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=FWD_I --dt=s32 --tag=axb --alg=linear --alpha=-0.25 --beta=-0.25 --inplace=true 3x17x2x5x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=FWD_I --dt=s32 --tag=axb --alg=relu --alpha=0.25 --beta=0 --inplace=true 5x16x3

./$bn --eltwise --engine=gpu  --skip-impl=ref --dt=f64 --alg=log --alpha=0 --beta=0 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f32 --alg=exp --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=exp --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=exp_dst --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=gelu_erf --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=gelu_tanh --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=logistic --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref  --tag=axb --alg=logistic --alpha=0 --beta=0 2x32x3x2
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --tag=axb --alg=logistic --alpha=0 --beta=0 2x32x3x2
./$bn  --eltwise --engine=gpu --skip-impl=ref --dt=bf16 --tag=aBx8b --alg=soft_relu --alpha=1 --beta=0 --inplace=true 3x7x3x2
./$bn  --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=soft_relu --alpha=1 --beta=0 --inplace=true 3x7x3x2
./$bn  --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=mish --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --alg=logistic --alpha=0 --beta=0 --inplace=true 16x64x1x1
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --tag=aBx8b --alg=exp_dst --alpha=0 --beta=0 --inplace=true 3x7x3x2
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=BWD_D --dt=f64 --tag=aBx8b --alg=soft_relu --alpha=-1 --beta=0 32x5x2x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=BWD_D  --tag=aBx8b --alg=soft_relu --alpha=-1 --beta=0 32x5x2x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --tag=ABx16a16b --alg=soft_relu --alpha=0.5 --beta=0 --inplace=true 3x17x2x5x3
./$bn  --zeropad --engine=gpu --skip-impl=ref --dt=f64 --tag=ABx16a16b 3x17x2x5x3 


./$bn  --eltwise --engine=gpu --skip-impl=ref --dt=f64 --tag=ABx16a16b --alg=pow --alpha=-0.25 --beta=0.5 --inplace=true 3x17x2x5x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=BWD_D --dt=f64 --alg=logistic_dst --alpha=0 --beta=0 --inplace=true 5x16x3
./$bn  --eltwise --engine=gpu --skip-impl=ref --dir=BWD_D --dt=f64 --tag=aBx8b --alg=gelu_erf --alpha=0 --beta=0 3x17x2x5x3
./$bn --eltwise --engine=gpu --skip-impl=ref --dir=BWD_D --dt=f64 --tag=aBx16b --alg=pow --alpha=-0.25 --beta=1.5 16x64x1x1
./$bn --eltwise --engine=gpu --skip-impl=ref --dt=f64 --tag=aBx16b --alg=hardswish --alpha=0.2 --beta=0.5 --inplace=true 3x17x2x5x3
./$bn --eltwise --engine=gpu --skip-impl=ref  --tag=aBx16b --alg=hardswish --alpha=0.2 --beta=0.5 --inplace=true 3x17x2x5x3

exit;
fi




echo "ERROR: command not found"
usage;




#export OMP_NUM_THREAD$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2S=1
#./$bn --deconv --engine=gpu --batch=test_deconv_gpu

#./$bn --deconv --engine=gpu --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n
#./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n

#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab 2x2:2x2


