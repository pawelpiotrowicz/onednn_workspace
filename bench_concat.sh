#!/bin/bash
#bn="/oneDNN/build/tests/benchdnn/benchdnn"
bn="/libraries.performance.math.onednn/build/tests/benchdnn/benchdnn"


./$bn --concat --engine=gpu --batch=test_concat_gpu --batch=test_concat_ci
