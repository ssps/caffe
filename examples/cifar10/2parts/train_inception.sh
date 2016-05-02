#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/cifar10/2parts/inception_train_val.prototxt 2
python scripts/duplicate.py examples/cifar10/2parts/inception_solver.prototxt 2
pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/2parts/inception_solver.prototxt --ps_config=examples/cifar10/2parts/ps_config --machinefile=examples/cifar10/2parts/machinefile --worker_id=%n"
