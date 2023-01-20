#!/bin/bash
bn="/oneDNN/build/tests/benchdnn/benchdnn"
export OMP_NUM_THREADS=1
./$bn --concat --engine=gpu --batch=test_concat_gpu


#./$bn --concat --engine=gpu -v5 --attr-scales=msrc0:common:2* --stag=ab:ab --dtag=ab 2x2:2x2


