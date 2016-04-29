#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/8parts/inception_train_val.prototxt 8
python scripts/duplicate.py examples/imagenet/8parts/inception_solver.prototxt 8
pdsh -R ssh -w ^examples/imagenet/8parts/machinefile "cd /users/hengganc/tank/geeps/apps/caffe && ./build/tools/caffe_geeps train --solver=examples/imagenet/8parts/inception_solver.prototxt --ps_config=examples/imagenet/8parts/ps_config_inception --worker_id=%n"
