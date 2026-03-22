from collections import defaultdict
from pathlib import Path
import os
import pickle
import torch
import numpy as np
import torch.utils.data as torch_data
# from data_loader_odo import data_loader
from ..utils import common_utils


from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from .processor.point_processor import MapsGenerator
from pcdet.utils import calibration_kitti
from .kitti import kitti_utils

import sys
from pathlib import Path
from skimage import io

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class DatasetTemplate(torch_data.Dataset):
    # openpcdet的kKittiDataset(DatasetTemplate)通过继承DatasetTemplate，这里直接修改DatasetTemplate作为加载器
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg # 数据集配置文件
        self.training = training       # 训练模式
        self.class_names = class_names # 类别
        self.logger = logger           # 日志
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32) # [-50.0, -25.0, -2.5, 50.0, 25.0, 1.5]
        # 创建点云特征编码网络
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        # 创建数据增强器类
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        # 创建数据预处理器类 
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )
        # 创建可见性地图生成类 
        self.map_generator =  MapsGenerator()

        self.grid_size = self.data_processor.grid_size      # 网格数量 = 点云范围 / 体素大小
        self.voxel_size = self.data_processor.voxel_size    # 体素大小
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        self._merge_his_frame = self.dataset_cfg.POINT_MERGE_HISFRAME # 是否需要叠加历史帧
        print(f'Use <{self._merge_his_frame}> his frames to training')

        # 导入数据集路径
        self.root = Path(self.dataset_cfg.DATA_LOAD_PATH)  
        self.gt_obser_bin_root = self.root / 'kitti_vis_8'
        self.gt_dense_lab_root = self.root / 'dense_label'
        self.gt_sparse_img_root= self.root / 'sparse_label'
        self.image_root= self.root / 'kitti_image'
        self.point_root= self.root / 'kitti_odo'

        if training == True:
            self.split = "train"
            file_name = self.dataset_cfg.DATA_TRAIN_PATH
        else:
            self.split = "test"
            file_name = self.dataset_cfg.DATA_TEST_PATH
        self.files_seq = []

        open_file = open(file_name, "rb")           # 打开sample_test.pkl文件
        self.files_seq = pickle.load(open_file)     # 将pkl的内容读取出来
        open_file.close()
        # print(len(self.files_seq)                 # 后面处理中有 point_path = self.files_seq[index].rstrip()

    @property
    def mode(self):

        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        """
        合并所有的iters到一个epoch中
        """
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def get_frame_point(self, seq, index):
        """
        读取当前点云 ----'kitti_odo'
        """
        f_file = self.point_root / seq / ('%s.bin' % index)
        points = np.fromfile(str(f_file), dtype=np.float32, count=-1).reshape([-1, 4]) 
        return points 

    def merge_history_point(self, points, seq, index, interval=2, num = 1):
        """
        读取历史点云 ----str str int
        """
        calib_path = self.point_root / seq / 'calib.txt'
        pose_path = self.point_root / seq / 'poses.txt'
        pose_mtx_lib = np.loadtxt(pose_path, dtype=np.float32) # 读取为[行数,12]
        pose_mtx_c = pose_mtx_lib[int(index)].reshape(3, 4)
        calib_lines = [line.rstrip('\n') for line in open(calib_path, 'r')]
        for calib_line in calib_lines:
            if 'Tr' in calib_line:
                tr_mtx = calib_line.split(' ')[1:]
                tr_mtx = np.array(tr_mtx, dtype='float').reshape(3,4) 
        # 坐标变换
        combined_mtx = pose_mtx_c @ np.vstack([tr_mtx, [0, 0, 0, 1]])  # [4x4] 综合变换矩阵
        # 分别处理xyz和ref
        curr_point = points[:,0:3] 
        point_ref = points[:, 3] 

        lidar_global = np.hstack((curr_point, np.ones((curr_point.shape[0], 1)))) @ combined_mtx.T  # [N, 4]
        combined_lidar = lidar_global[:, :3] # 转换到原点坐标
        his_index = index
        # 堆叠 num 帧
        for _ in range(num):  
            if int(his_index) >= interval:    # 比较 a 的整数部分和 b
                result = int(his_index) - interval  # 计算 a - b 并转换为字符串
                his_index = f"{result:0{len(his_index)}d}"

                point_path = self.point_root / seq / ('%s.bin' % his_index)
                read_points = np.fromfile(point_path, dtype=np.float32, count=-1).reshape(-1, 4)[:,0:4] # (124823, 3)
                his_points = read_points[:,0:3]
                point_ref = np.concatenate((point_ref, read_points[:, 3]), axis=0)

                pose_mtx_his = pose_mtx_lib[result].reshape(3, 4)
                combined_mtx = pose_mtx_his @ np.vstack([tr_mtx, [0, 0, 0, 1]])  # [4x4] 综合变换矩阵
                lidar_hst_global = np.hstack((his_points, np.ones((his_points.shape[0], 1)))) @ combined_mtx.T  # [N, 4]
                lidar_hst_global = lidar_hst_global[:, :3]

                combined_lidar = np.vstack([combined_lidar,lidar_hst_global])
            else:
                combined_lidar = np.vstack([combined_lidar,lidar_global[:, :3]])
                point_ref = np.concatenate((point_ref, points[:, 3]), axis=0)
        # 矩阵逆变换
        pose_mtx_4x4 = np.vstack((pose_mtx_c, [0, 0, 0, 1]))
        tr_mtx_4x4 = np.vstack((tr_mtx, [0, 0, 0, 1]))
        pose_mtx_inv = np.linalg.inv(pose_mtx_4x4)
        tr_mtx_inv = np.linalg.inv(tr_mtx_4x4)
        # 恢复到原始 LiDAR 坐标
        lidar_global_h = np.c_[combined_lidar, np.ones(combined_lidar.shape[0])]  # 确保齐次坐标
        lidar_ego_recovered = lidar_global_h @ pose_mtx_inv.T
        curr_lidar_recovered = lidar_ego_recovered @ tr_mtx_inv.T
        # 去掉齐次坐标，恢复到 [x, y, z]
        curr_lidar_recovered = curr_lidar_recovered[:, 0:3]
        point_ref = point_ref.reshape(-1, 1)
        # merged_points = np.concatenate([curr_lidar_recovered, point_ref, np.zeros([curr_lidar_recovered.shape[0], 1])], axis=-1) # 4->5
        merged_points = np.concatenate([curr_lidar_recovered, point_ref], axis=-1) # 4->5

        return merged_points 

    def get_dense_gt_lab(self, seq, index):
        """
        读取密集的地图的ground truth .bin  --> seg_gt
        """
        f_file = self.gt_dense_lab_root / seq / ('%s.bin' % index) # f_file dense_label/02/000006.bin
        assert f_file.exists()
        return np.fromfile(f_file, dtype=np.float32).reshape(500, 1000, 1)
    
    def get_obser_bin(self, seq, index):
        """
        读取visibility map .bin --> obser
        """
        f_file = self.gt_obser_bin_root / seq / ('%06d.bin' % int(index))
        assert f_file.exists()
        return np.fromfile(str(f_file), dtype=np.float32, count=-1).reshape(2, 500, 1000, 1) 

    def get_sparse_gt_lab(self, seq, index):
        """
        读取离散标签 .png ---- sparse_label
        """
        f_file = self.gt_sparse_img_root / seq / ('%06d.bin' % int(index))
        # print(self.gt_sparse_img_root,seq,index)
        assert f_file.exists()
        return np.fromfile(f_file, dtype=np.int8).reshape(500, 1000, 1)
        # return np.array(io.imread(f_file), dtype=np.int8).reshape(500, 1000, 1)

    def get_image(self, seq, index):
        img_file = self.image_root / seq / 'image_2' / ('%06d.png' % int(index))
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0

        return image # (370, 1226, 3) 


    def __len__(self):
        """
        __len__ 返回pkl文件元素的长度
        """
        return len(self.files_seq) 

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns:

        """
        data_dict = {}
        # 点云路径
        point_file = self.files_seq[index].rstrip()  # rstrip删除末尾空格 00/000009 
        seq = point_file.split('/')[-2] # 返回 00, 01, 02 ..
        idx = point_file.split('/')[-1]  # 返回路径的文件名部分，不包括后缀  "00/velodyne/000000.bin"  -> 000000
        data_dict['frame_id'] = seq + idx  # 01000001 
        
        # 1、加载当帧点云
        points = self.get_frame_point(seq, idx) 
        ref_trans = self.map_generator.Ref_Trans_inGrid(points[:,0:3]) #  [1, 2, 500, 1000, 4]
        # 这里没经过mask过滤点云处理是应为Ref_Trans_inGrid有过滤处理了

        # 2、获取历史帧
        if self._merge_his_frame != 0:
            points = self.merge_history_point(points, seq, idx, interval = 1, num = self._merge_his_frame)

        mask = common_utils.mask_points_by_range(points, self.point_cloud_range) # mask除去范围外的点
        points = points[mask]
        points = np.concatenate([points, np.zeros([points.shape[0], 1])], axis=-1) # 4->5
        data_dict["points"] = points

        # 3、加载点云对应标签
        seg_gt = self.get_dense_gt_lab(seq, idx)  # f_file = self.gt_dense_img_root / seq / ('%s.png' % index) 
        # seg_gt = self.get_sparse_gt_lab(seq, idx)
        data_dict['labels_seg'] = seg_gt
        # 读取点云对应的可见性地图[0]和二值地图[1]
        # obser =  np.zeros((500, 1000, 1)) # 暂时用全0填充
        obser =  self.get_obser_bin(seq, idx)
        data_dict['observations'] = obser[0] # (500, 1000, 1)
        data_dict['binarymap'] = obser[1] # (500, 1000, 1) ndarray

        data_dict['reflections'] = ref_trans[0, 0].numpy() # [500, 1000, 8] Tensor
        data_dict['transmissions'] = ref_trans[0, 1].numpy() # [500, 1000, 8]

        data_dict['ref_trans'] = ref_trans.squeeze(0) # cpu [2, 500, 1000, 8]
        # 特征mask
        data_dict['voxlmask'] = np.ones((500, 1000, 8), dtype=int)
        # 对数据进行处理！
        data_dict = self.prepare_data(data_dict) 
        # 调整dense_label的通道格式
        seg_gt = np.transpose(data_dict['labels_seg'], (2, 0, 1))  # 1, 500, 1000
        data_dict['labels_seg'] = seg_gt
        # # 调整visibility map的通道格式
        # obser = np.transpose(data_dict['observations'], (2, 0, 1))
        # data_dict['observations'] = obser / 255  # normalization

        return data_dict

    def set_lidar_aug_matrix(self, data_dict):
        """
            Get lidar augment matrix (4 x 4), which are used to recover orig point coordinates.
        """
        lidar_aug_matrix = np.eye(4)
        if 'flip_y' in data_dict.keys():
            flip_x = data_dict['flip_x']
            flip_y = data_dict['flip_y']
            if flip_x:
                lidar_aug_matrix[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
            if flip_y:
                lidar_aug_matrix[:3,:3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
        if 'noise_rot' in data_dict.keys():
            noise_rot = data_dict['noise_rot']
            lidar_aug_matrix[:3,:3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ lidar_aug_matrix[:3,:3]
        if 'noise_scale' in data_dict.keys():
            noise_scale = data_dict['noise_scale']
            lidar_aug_matrix[:3,:3] *= noise_scale
        if 'noise_translate' in data_dict.keys():
            noise_translate = data_dict['noise_translate']
            lidar_aug_matrix[:3,3:4] = noise_translate.T
        data_dict['lidar_aug_matrix'] = lidar_aug_matrix
        return data_dict


    def prepare_data(self, data_dict):
        """
        接受统一坐标系下的数据字典(points),进行数据筛选,数据预处理,包括数据增强,点云编码等
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # 使用旋转平移等堆gt_seg, points, observations进行一致化数据增强
        if self.training:
            data_dict = self.data_augmentor.forward(data_dict=data_dict)
        data_dict = self.set_lidar_aug_matrix(data_dict) # 计算lidar变换增强后的矩阵，用于图像变换对齐

        data_dict = self.data_processor.forward(
            data_dict = data_dict
        )

        if self.training and len(data_dict['voxel_coords']) == 0:
            new_index = np.random.randint(self.__len__())
            print('Error frame id: ', data_dict['frame_id'])
            return self.__getitem__(new_index)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        # data_dict.pop('points_sp', None)
        data_dict.pop('indices', None)
        data_dict.pop('origins', None)

        ret['frame_id'] = []
        for key, val in data_dict.items():
            # print('collate: ', key)
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'dense_point']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['img_process_infos']:
                    ret[key] = np.array(val, dtype=object)
                elif key == 'frame_id':
                    ret[key].append(data_dict['frame_id'])
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        ret['batch_size'] = batch_size

        return ret

