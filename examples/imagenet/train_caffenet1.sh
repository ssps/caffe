#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/1part/caffenet_train_val.prototxt 1
python scripts/duplicate.py examples/imagenet/1part/caffenet_solver.prototxt 1
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/imagenet/train_caffenet1.sh $1/.
cp examples/imagenet/1part/caffenet_train_val.prototxt.template $1/.
cp examples/imagenet/1part/caffenet_solver.prototxt.template $1/.
cp examples/imagenet/1part/machinefile $1/.
cp examples/imagenet/1part/ps_config_caffenet $1/.
mpirun -machinefile examples/imagenet/1part/machinefile ./build/tools/caffe_mpi train --solver=examples/imagenet/1part/caffenet_solver.prototxt --ps_config=examples/imagenet/1part/ps_config_caffenet 2>&1 | tee $1/output.txt
