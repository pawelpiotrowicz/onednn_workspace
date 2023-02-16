#!/bin/bash
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
export DNNL_VERBOSE=3
usage() {

echo "exit() : --all --postop64 --simple32 --simple64 --small --fail --conv-all"
exit
}

if [ "$#" -ne 1 ]; then
usage;
fi

if [ "$1" == "--conv-all" ]; then
./$bn --conv --engine=gpu --batch=test_conv_gpu
exit;
fi

if [ "$1" == "--all" ]; then
./$bn --deconv --engine=gpu --batch=test_deconv_gpu
exit;
fi

if [ "$1" == "--small" ]; then
./$bn --deconv --engine=gpu --dir=BWD_WB --cfg=f64 mb96ic96ih28oc192oh28kh1ph0n"googlenet_v1:inception_3a/3x3_reduce"
./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f64+add:f64:per_dim_01+linear:0.5:1.5+mul:f64:per_dim_0+add:f64:per_oc+add:f64:per_dim_01+relu:0.5 g1ic96iw55oc3ow227kw11sw4pw0n"alexnet:deconv1"

exit;
fi



if [ "$1" == "--postop64" ]; then

./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2

exit;
fi



if [ "$1" == "--simple32" ]; then
./$bn --deconv --engine=gpu --cfg=f32 --mb=2 --dir=FWD_B ic8iw5oc8ow2kw3pw3dw2n"deconv1d:1"
exit;
fi



if [ "$1" == "--simple64" ]; then
./$bn --deconv --engine=gpu --cfg=f64 --mb=2 --dir=FWD_B ic8iw5oc8ow2kw3pw3dw2n"deconv1d:1"
exit;
fi


if [ "$1" == "--fail" ]; then
./$bn --deconv --engine=gpu --dir=BWD_WB --cfg=bf16f32bf16 mb16ic128ih3oc48oh7kh5ph2dh1n"5b/5x5"
./$bn --deconv --engine=gpu --dir=BWD_WB --cfg=f64f64f64 mb16ic128ih3oc48oh7kh5ph2dh1n"5b/5x5"
#./$bn --deconv --engine=gpu --dir=BWD_WB --cfg=f64f32bf16 mb16ic128ih3oc48oh7kh5ph2dh1n"5b/5x5"

exit;
fi




echo "ERROR: command not found"
usage;




#export OMP_NUM_THREAD$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2S=1
#./$bn --deconv --engine=gpu --batch=test_deconv_gpu

#./$bn --deconv --engine=gpu --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n
#./$bn --deconv --engine=gpu --cfg=f64 --attr-post-ops=sum:0.5+add:f32+add:u8:per_dim_01+linear:0.5:1.5+mul:f32:per_dim_0+add:s8:per_oc+add:f32:per_dim_01+relu:0.5 ic8iw5oc8ow2kw3pw3dw2n

#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab 2x2:2x2


