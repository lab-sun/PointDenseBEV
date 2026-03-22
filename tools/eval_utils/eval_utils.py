import pickle
import time
import math
import numpy as np
import torch
import tqdm
# import Path
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import sys
import time
import os
import cv2
from PIL import Image


color_map_12={
  "0" : [255, 255, 255],
  "1": [0, 0, 255],
  "11": [255, 120, 60],
  "3": [245, 230, 100],
  "4": [200, 40, 50],
  "5": [255, 0, 255],
  "8": [255, 200, 20],
  "10": [0, 175, 0],
  "6": [75, 0, 75],
  "7": [75, 0, 175],
  "2": [30, 30, 255],
  "12": [150, 240, 80],
  "9": [135,60,0]
}
class_name_12 = ["vehicle", "person", "two-wheel", "rider", "road", "sidewalk", "otherground", "building",
                "object", "vegetation", "trunk", "terrain"]
color_map_14 ={
"0": [255, 255, 255],
"1": [112, 128, 144],
"2": [220, 20, 60],
"3": [255, 127, 80],
"4": [255, 158, 0],
"5": [233, 150, 70],
# "6": [255, 61, 99],
"6": [0, 0, 230],
"7": [47, 79, 79],
# "9": [255, 140, 0],
"8": [255, 99, 71],
"9": [0, 207, 191],
"10": [175, 0, 75],
"11": [75, 0, 75],
"12": [112, 180, 60],
"13": [222, 184, 135],
"14": [0, 175, 0]
}
class_name_14 = ["barrier", "two_wheel", "bus", "car", "work_vehicle", "pedestrian", "traffic_cone", "truck", "drive", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]


def id_to_rgb(pred_id, class_num):
    color_map = color_map_12 if class_num == 12 else color_map_14
    shape = list(pred_id.shape)[:2]
    shape.append(3)
    rgb = np.zeros(shape, dtype=np.uint8) + 255
    for i in range(0, class_num):
        mask = pred_id[:, :, 0] == i
        # print(mask.shape)
        if mask.sum() == 0:
            continue
        rgb[mask] = np.array(color_map[str(i + 1)])
    return rgb.astype(np.uint8)


def get_iou(pred, gt, number, intersect, union, index, logger, result_dir, time_stamp, n_classes):
    # total_miou = 0.0
    # class_name = ["vehicle", "person", "two-wheel", "rider", "road", "sidewalk", "otherground", "building", "object",
    #               "vegetation", "trunk", "terrain"]
    # iou_list_sum = torch.zeros([n_classes])

    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        for j in range(n_classes):
            match = (pred_tmp == j).int() + (gt_tmp == j).int()
            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()
            intersect[j] += it
            union[j] += un

    return intersect, union


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
        metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()
    start_time = time.time()
    n_val = len(dataloader)
    with tqdm.tqdm(total=n_val, desc='Val', unit='batch', leave=False) as pbar:
        class_num = 12 if cfg.DATA_CONFIG.DATASET == 'DatasetTemplate' else 14
        print("[DEBUG] class_num",class_num)
        intersection = torch.zeros(class_num)
        union = torch.zeros(class_num)
        for i, batch_dict in enumerate(dataloader):

            load_data_to_gpu(batch_dict)

            with torch.no_grad():
                pred_dict = model(batch_dict)
            torch.set_printoptions(profile="full")
            batch_size, c, h, w = pred_dict["prediction"].size()
            dense_gt = pred_dict["labels_seg"]  # 0 to 12 # 'torch.Tensor' torch.Size([1, 1, 500, 1000])
            observation = pred_dict["observations"] # torch.Size([1, 500, 1000, 1])
            binarymap = pred_dict['binarymap'] # torch.Size([1, 500, 1000, 1])
            
            pred = pred_dict["prediction"][:, :class_num, :, :] # torch.Size([1, 12, 500, 1000])
            pred = torch.argmax(pred, dim=1, keepdim=True)  # 2 1 500 1000  # torch.Size([1, 1, 500, 1000])
           
            ######################################################################################
            # 根据可见性调整预测区域，只评估可见部分和有lab部分的预测情况
            no_obser_mask = binarymap <= 0 # torch.Size([1, 500, 1000, 1])
            no_obser_mask = no_obser_mask.permute(0, 3, 1, 2) # 修改了通道顺序[1, 500, 1000, 1] -> [1, 1, 500, 1000]
            pred[no_obser_mask] = -1  # torch.Size([1, 1, 500, 1000])
            
            no_gt_mask = dense_gt.view(batch_size, 1, h, w) == 0
            pred[no_gt_mask] = -1
            dense_gt[no_obser_mask] = 0  # torch.Size([1, 1, 500, 1000])
            dense_gt = dense_gt - 1  # from -1 to 11

            # # 没有可见性图暂时只用一下指令
            # dense_gt = dense_gt - 1  # from -1 to 11
            #####################################################################################

            # save pred as img # 保存图片#########################################################
            pred_img = batch_dict['fullprediction'][:, :class_num, :, :]
            pred_img = torch.argmax(pred_img, dim=1, keepdim=True)
            pred_img[no_obser_mask] = -1 
            prediction_save_path = result_dir / "eval_segmentation_full/"
            prediction_save_path.mkdir(parents=True, exist_ok=True)

            for batch_index in range(batch_size):
                # img_path = prediction_save_path / ('%08d.png' % int(pred_dict['frame_id'][0][batch_index]))
                frame_id = str(pred_dict['frame_id'][0][batch_index])
                if frame_id.endswith('.pcd'):
                    frame_id = frame_id[:-4]
                safe_name = frame_id.replace(os.sep, '_').replace('\\', '_')
                img_path = prediction_save_path / f'{safe_name}.png'
                cls_id = pred_img[batch_index].cpu().numpy().astype(np.uint8)
                cls_id = np.transpose(cls_id, (1, 2, 0))
                rgb = id_to_rgb(cls_id, class_num)
                rgb = Image.fromarray(rgb)
                rgb.save(str(img_path))
            #####################################################################################



            intersection, union = get_iou(pred, dense_gt, i + 1, intersection, union, n_val * batch_size, logger,
                                          result_dir, timestamp, n_classes = class_num)
            pbar.update()

        class_list = []
        iou = []
        ret_dict = {}
        class_name = class_name_12 if class_num == 12 else class_name_14


        file = open(result_dir / ("eval_%s.txt" % epoch_id), 'a')
        for k in range(len(intersection)):
            if union[k] != 0:
                class_list.append(class_name[k])
                iou.append(intersection[k] / union[k])
            else:
                continue

        miou = ((sum(iou)) / (len(iou)))
        iou = torch.Tensor(iou)
        file.write("******************eval_whole_dataset******************\n")
        print("*******************eval_result whole dataset**********************:\n")
        for j, class_index in enumerate(class_list):
            logger.info('iou_%s: %f' % (class_index, iou[j]))
            file.write('iou_%s: %f' % (class_index, iou[j]) + '\n')
            ret_dict['iou_%s' % class_index] = iou[j]
        logger.info('miou: %f' % (miou))
        file.write('miou: %f' % (miou) + '\n')
        ret_dict['miou']= miou
        file.write("****************************************************\n")
        print("****************************************************")
        file.close()

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
