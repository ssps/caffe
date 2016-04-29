#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/1part/caffenet_train_val.prototxt 1
python scripts/duplicate.py examples/imagenet/1part/caffenet_solver.prototxt 1
pdsh -R ssh -w ^examples/imagenet/1part/machinefile "cd /users/hengganc/tank/geeps/apps/caffe && ./build/tools/caffe_geeps train --solver=examples/imagenet/1part/caffenet_solver.prototxt --ps_config=examples/imagenet/1part/ps_config_caffenet --worker_id=%n"
