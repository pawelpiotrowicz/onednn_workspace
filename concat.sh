#!/bin/bash
#bn="/oneDNN/build/tests/benchdnn/benchdnn"
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"
#./$bn --concat --engine=gpu --batch=test_concat_gpu

if [ -d "cldump" ]; then
  rm -rf cldump
fi
#export DNNL_JIT_DUMP=1
export LD_PRELOAD=`pwd`/libcldump1.so
export DNNL_VERBOSE=3
#./$bn --concat --engine=gpu -v5 --stag=ab:ab --dtag=ab  2x2:2x2
#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab  2x2:2x2
#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab --skip-impl=ref 2x2:2x2
#./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:2*+msrc1:common:2*+dst:common:2* --stag=ab:ab --dtag=ab --skip-impl=ref 2x2:2x2
#./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:2*+msrc1:common:2*+msrc2:common:2*+dst:common:2* --stag=ab:ab:ab --dtag=ab --skip-impl=ref 2x2:2x2:2x2

#######
#./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:1*+msrc1:common:1*+msrc2:common:1*+dst:common:2* --stag=ab:ab:ab --dtag=ab --skip-impl=ref 2x2:2x2:2x2
#./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:1*+msrc1:common:1*+msrc2:common:1*+dst:common:2* --stag=ab:ab:ab --dtag=ab  2x2:2x2:2x2
##########
#./$bn --concat --engine=gpu -v5  --attr-scales=dst:common:2* --stag=ab:ab:ab --dtag=ab --skip-impl=ref 2x2:2x2:2x2
#./$bn --concat --engine=gpu -v5  --attr-scales=dst:common:2* --stag=ab:ab:ab --dtag=ab  2x2:2x2:2x2

echo "=================== AFTER  ==================="
#./$bn --concat --engine=cpu -v100  --attr-scales=dst:common:2* --stag=ab:ab:ab --dtag=ab  2x2:2x2:2x2

### gen9_concat ====
#./$bn --concat --engine=gpu --allow-enum-tags-only=false -v5 --attr-scales=msrc0:common:1*+msrc1:common:1*+msrc2:common:1*+dst:common:2* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#./$bn --concat --engine=gpu --allow-enum-tags-only=false -v5 --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3*+dst:common:3* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#end gen9_cocnat 

echo "=================== CPU  ==================="
#./$bn --concat --engine=cpu --allow-enum-tags-only=false -v100 --attr-scales=msrc0:common:1*+msrc1:common:1*+msrc2:common:1*+dst:common:2* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2

########### TEN ####
#./$bn --concat --engine=gpu --allow-enum-tags-only=false -v5 --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2

#./$bn --concat --engine=gpu --allow-enum-tags-only=false --sdt=s8 --ddt=s32 --stag=ab:ab:ab --dtag=aB2b --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3* 2x2:2x2:2x2

#./$bn --concat --engine=gpu --allow-enum-tags-only=false --sdt=s8 --ddt=s32 --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2

#./$bn --concat --engine=gpu --allow-enum-tags-only=false --sdt=s32 --ddt=s32 --stag=ab:ab:ab --dtag=aB2b --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3* 2x2:2x2:2x2

./$bn --concat --engine=gpu --sdt=s32,s8,u8,bf16,f16 --ddt=s32,s8,u8,bf16,f16 --allow-enum-tags-only=false -v5 --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#./$bn --concat --engine=gpu --sdt=s32,s8,u8,bf16,f16 --ddt=s32,s8,u8,bf16,f16 --allow-enum-tags-only=false -v5 --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#./$bn --concat --engine=gpu --sdt=s32 --ddt=s8 --allow-enum-tags-only=false -v5 --attr-scales=msrc0:common:3*+msrc1:common:3*+msrc2:common:3* --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#./$bn --concat --engine=gpu --sdt=s32 --ddt=s8 --allow-enum-tags-only=false -v5  --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
######################

#./$bn --concat --engine=gpu --allow-enum-tags-only=false -v5 --stag=ab:ab:ab --dtag=aB2b 2x2:2x2:2x2
#./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:1*+msrc1:common:1*+msrc2:common:1*+dst:common:2* --stag=ab:ab:ab  --dtag=aB2b  --allow-enum-tags-only=false 2x2:2x2:2x2
