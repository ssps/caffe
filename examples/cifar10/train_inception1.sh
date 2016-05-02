#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/1part/inception_train_val.prototxt 1
python scripts/duplicate.py examples/cifar10/1part/inception_solver.prototxt 1
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/cifar10/train_inception1.sh $1/.
cp examples/cifar10/1part/inception_train_val.prototxt.template $1/.
cp examples/cifar10/1part/inception_solver.prototxt.template $1/.
cp examples/cifar10/1part/machinefile $1/.
cp examples/cifar10/1part/ps_config $1/.
mpirun -machinefile examples/cifar10/1part/machinefile ./build/tools/caffe_mpi train --solver=examples/cifar10/1part/inception_solver.prototxt --ps_config=examples/cifar10/1part/ps_config 2>&1 | tee $1/output.txt
