CUDA_VISIBLE_DEVICES=6 python ./tools/run_train.py \
--batch_size 1 \
--epochs 24 \
--cfg_file './tools/cfgs/kitti_models/test_pillarnet_128.yaml' \
--extra_tag 'test' \
