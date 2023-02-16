#!/bin/bash
#bn="/oneDNN/build/tests/benchdnn/benchdnn"
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
#export DNNL_JIT_DUMP=1
#export LD_PRELOAD=`pwd`/libcldump1.so

#./$bn  --concat --engine=gpu --mode=c --batch=batch_files/concat_pvc_kpi.txt




#./$bn  --concat --engine=gpu --mode=po --batch=batch_files/concat_pvc_kpi.txt
# cat XXX.log | grep "^perf," | cut -d, -f7



#export DNNL_VERBOSE=2

#./$bn --concat --engine=gpu --allow-enum-tags-only=false --mode=c --sdt=bf16 --ddt=bf16 --stag=acdeb:aBcde16b --dtag=aBcde16b 2x320x8x8x8:2x320x8x8x8
#./$bn  --concat --engine=gpu --allow-enum-tags-only=false --mode=c --sdt=bf16 --ddt=bf16 --stag=acdeb:aBcde16b --dtag=aBcde16b 2x64x64x64x64:2x64x64x64x64

#./$bn --concat --engine=gpu --allow-enum-tags-only=false --mode=c --sdt=bf16 --ddt=bf16 --stag=acdeb:aBcde16b --dtag=aBcde16b 2x64x64x64x64:2x64x64x64x64


#./$bn --concat --engine=gpu --batch=test_concat_gpu --batch=test_concat_ci


##### ISSUE PERFORMANCE ####

./$bn  --concat --engine=gpu --allow-enum-tags-only=0 --sdt=f32 --ddt=f32 --stag=abc:abc:abc --dtag=abc --mode=po --axis=2 4x128x2:4x128x1:4x128x1  | grep "^perf," | cut -d, -f7

