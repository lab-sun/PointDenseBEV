CUDA_VISIBLE_DEVICES=0 python ./tools/run_test.py \
--batch_size 1 \
--start_epoch 0 \
--cfg_file './tools/cfgs/kitti_models/densepillar.yaml' \
--extra_tag 'test_his0' \
--ckpt_dir '/home/Projects/DensePillar/logs/model/kitti_models/densepillar/test_his0/ckpt/' 
