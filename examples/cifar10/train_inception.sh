#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/8parts/inception_train_val.prototxt 8
python scripts/duplicate.py examples/cifar10/8parts/inception_solver.prototxt 8
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/cifar10/train_inception1.sh $1/.
cp examples/cifar10/8parts/inception_train_val.prototxt.template $1/.
cp examples/cifar10/8parts/inception_solver.prototxt.template $1/.
cp examples/cifar10/8parts/machinefile $1/.
cp examples/cifar10/8parts/ps_config $1/.
mpirun -machinefile examples/cifar10/8parts/machinefile ./build/tools/caffe_mpi train --solver=examples/cifar10/8parts/inception_solver.prototxt --ps_config=examples/cifar10/8parts/ps_config 2>&1 | tee $1/output.txt
