#!/bin/bash
bn="/oneDNN/build/tests/benchdnn/benchdnn"
#./$bn --concat --engine=gpu --batch=test_concat_gpu

export DNNL_VERBOSE=1
./$bn --concat --engine=gpu -v5 --stag=ab:ab --dtag=ab  2x2:2x2
./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab  2x2:2x2
./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab --skip-impl=ref 2x2:2x2
./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:2*+msrc1:common:2*+dst:common:2* --stag=ab:ab --dtag=ab --skip-impl=ref 2x2:2x2
./$bn --concat --engine=gpu -v5  --attr-scales=msrc0:common:2*+msrc1:common:2*+msrc2:common:2*+dst:common:2* --stag=ab:ab:ab --dtag=ab --skip-impl=ref 2x2:2x2:2x2


