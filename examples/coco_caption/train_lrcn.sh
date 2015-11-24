#!/usr/bin/env bash

python scripts/duplicate.py examples/coco_caption/8parts/lrcn.prototxt 8
python scripts/duplicate.py examples/coco_caption/8parts/lrcn_solver.prototxt 8
mkdir $1
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff

mpirun -machinefile examples/coco_caption/8parts/machinefile ./build/tools/caffe_mpi train --solver=./examples/coco_caption/8parts/lrcn_solver.prototxt --weights=./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --ps_config=examples/coco_caption/8parts/ps_config 2>&1 | tee $1/output.txt
