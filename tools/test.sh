CUDA_VISIBLE_DEVICES=7 python ./tools/run_test.py \
--batch_size 1 \
--start_epoch 10 \
--cfg_file './tools/cfgs/kitti_models/test_pillarnet_128.yaml' \
--extra_tag 'pretrain' \
