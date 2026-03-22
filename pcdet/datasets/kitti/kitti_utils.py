import numpy as np



def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # ((3, 4) -> (4, 4) LiDAR 到参考相机
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # ((3, 3) -> (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # ((3, 4) -> (4, 4)
    V2R = R0 @ V2C # : (4, 4) @ (4, 4) -> (4, 4) 计算 LiDAR 到 矫正相机
    P2 = calib.P2 # (3, 4)

    camera_intrinsics = P2[:3, :3] # (3, 3)
    C2L = np.linalg.inv(R0 @ V2C) # (4, 4) -> (4, 4)
    L2I = P2 @ V2R # (3, 4) @ (4, 4) -> (3, 4)
    return camera_intrinsics, C2L, L2I, V2R, P2