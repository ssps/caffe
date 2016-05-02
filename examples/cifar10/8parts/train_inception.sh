#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/cifar10/8parts/inception_train_val.prototxt 8
python scripts/duplicate.py examples/cifar10/8parts/inception_solver.prototxt 8
pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/8parts/inception_solver.prototxt --ps_config=examples/cifar10/8parts/ps_config --machinefile=examples/cifar10/8parts/machinefile --worker_id=%n"
