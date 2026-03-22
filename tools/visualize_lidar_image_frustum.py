import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_matrix4x4(path: str):
    p = Path(path)
    if p.suffix == ".npy":
        mat = np.load(str(p))
    else:
        mat = np.loadtxt(str(p), dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expect 4x4 matrix in {path}, got {mat.shape}")
    return mat


def apply_transform(points_xyz: np.ndarray, tf_4x4: np.ndarray):
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)], axis=1)
    out = points_h @ tf_4x4.T
    return out[:, :3]


def center_crop_and_img_aug_matrix(image: np.ndarray, final_h: int, final_w: int):
    """
    Align with DataProcessor.image_crop (test branch) + image_calibrate.
    resize=1.0, flip=False, rotate=0.
    """
    h, w = image.shape[:2]
    crop_h = h - final_h
    crop_w = (w - final_w) // 2
    crop_h = max(crop_h, 0)
    crop_w = max(crop_w, 0)
    crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)

    cropped = image[crop_h:crop_h + final_h, crop_w:crop_w + final_w]

    transform = np.eye(4, dtype=np.float64)
    transform[:2, 3] = -np.array(crop[:2], dtype=np.float64)
    return cropped, transform, crop


def read_kitti_calib(calib_path: Path):
    calib_all = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()], dtype=np.float64)

    p2 = np.eye(4, dtype=np.float64)
    tr = np.eye(4, dtype=np.float64)
    p2[:3, :4] = calib_all["P2"].reshape(3, 4)
    tr[:3, :4] = calib_all["Tr"].reshape(3, 4)
    return p2, tr


def project_lidar_to_image(points_xyz: np.ndarray, lidar2image: np.ndarray):
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)], axis=1)
    proj = points_h @ lidar2image.T
    depth = proj[:, 2]
    depth_safe = np.clip(depth, 1e-6, None)
    uv = proj[:, :2] / depth_safe[:, None]
    return uv, depth


def create_frustum(image_size, feature_size, dbound):
    i_h, i_w = image_size
    f_h, f_w = feature_size
    d_start, d_end, d_step = dbound

    ds = np.arange(d_start, d_end, d_step, dtype=np.float64)[:, None, None]
    ds = np.broadcast_to(ds, (ds.shape[0], f_h, f_w))
    xs = np.linspace(0, i_w - 1, f_w, dtype=np.float64)[None, None, :]
    xs = np.broadcast_to(xs, (ds.shape[0], f_h, f_w))
    ys = np.linspace(0, i_h - 1, f_h, dtype=np.float64)[None, :, None]
    ys = np.broadcast_to(ys, (ds.shape[0], f_h, f_w))
    frustum = np.stack([xs, ys, ds], axis=-1)  # (D, fH, fW, 3): (u, v, d)
    return frustum


def frustum_to_lidar(frustum, cam_intrinsics, camera2lidar, img_aug_matrix=None, lidar_aug_matrix=None):
    points = frustum.reshape(-1, 3)
    if img_aug_matrix is not None:
        # Match DepthLSSTransform: undo post image transform first.
        points = (np.linalg.inv(img_aug_matrix[:3, :3]) @ (points - img_aug_matrix[:3, 3][None, :]).T).T

    uv1 = np.stack([points[:, 0], points[:, 1], np.ones_like(points[:, 0])], axis=1)  # (u, v, 1)
    cam_xyz = (np.linalg.inv(cam_intrinsics) @ uv1.T).T * points[:, 2:3]  # (x, y, z) in camera

    rot = camera2lidar[:3, :3]
    trans = camera2lidar[:3, 3]
    lidar_xyz = (rot @ cam_xyz.T).T + trans[None, :]
    if lidar_aug_matrix is not None:
        # Match DepthLSSTransform: apply lidar augmentation in BEV frame.
        lidar_xyz = apply_transform(lidar_xyz, lidar_aug_matrix)
    return lidar_xyz


def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR->image projection and frustum->LiDAR backprojection.")
    parser.add_argument("--calib", type=str, required=True, help="KITTI calib.txt path")
    parser.add_argument("--lidar", type=str, required=True, help="KITTI velodyne .bin path")
    parser.add_argument("--image", type=str, required=True, help="KITTI image_2 .png path")
    parser.add_argument("--output", type=str, default="projection_debug.png", help="Output figure path")
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 1024], help="[H W] used by frustum")
    parser.add_argument("--feature-size", type=int, nargs=2, default=[40, 128], help="[fH fW] frustum feature size")
    parser.add_argument("--dbound", type=float, nargs=3, default=[1.0, 50.0, 0.25], help="[start end step]")
    parser.add_argument("--frustum-sample", type=int, default=10, help="Plot every N-th frustum point")
    parser.add_argument("--lidar-sample", type=int, default=1, help="Plot every N-th lidar point")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-50.0, 50.0], help="BEV x range")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-25.0, 25.0], help="BEV y range")
    parser.add_argument("--use-augment", action="store_true", help="Use img/lidar aug matrices like depth_lss.py")
    parser.add_argument("--img-aug-matrix", type=str, default=None, help="4x4 image aug matrix path (.txt or .npy)")
    parser.add_argument("--lidar-aug-matrix", type=str, default=None, help="4x4 lidar aug matrix path (.txt or .npy)")
    args = parser.parse_args()

    calib_path = Path(args.calib)
    lidar_path = Path(args.lidar)
    image_path = Path(args.image)
    output_path = Path(args.output)

    p2, tr = read_kitti_calib(calib_path)
    lidar2image = p2[:3, :] @ tr
    camera2lidar = np.linalg.inv(tr)
    cam_intrinsics = p2[:3, :3]

    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
    points_xyz = points[:, :3]
    if args.lidar_sample > 1:
        points_xyz = points_xyz[:: args.lidar_sample]

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image, crop_img_aug_matrix, _ = center_crop_and_img_aug_matrix(
        image_raw, args.image_size[0], args.image_size[1]
    )
    img_h, img_w = image.shape[:2]
    if (img_h, img_w) != (args.image_size[0], args.image_size[1]):
        print(
            f"[WARN] Requested image-size {tuple(args.image_size)} exceeds raw image size; "
            f"actual cropped size is {(img_h, img_w)}."
        )

    img_aug_matrix = crop_img_aug_matrix.copy()
    lidar_aug_matrix = np.eye(4, dtype=np.float64)
    if args.use_augment:
        if args.img_aug_matrix is not None:
            img_aug_matrix = load_matrix4x4(args.img_aug_matrix)
        if args.lidar_aug_matrix is not None:
            lidar_aug_matrix = load_matrix4x4(args.lidar_aug_matrix)
        else:
            print("[WARN] --use-augment enabled but --lidar-aug-matrix not provided, use identity.")

    points_xyz_bev = points_xyz
    points_xyz_for_proj = points_xyz
    if args.use_augment:
        # Simulate data_augmentor output points, then inverse in projection path (as in depth_lss.py).
        points_xyz_bev = apply_transform(points_xyz, lidar_aug_matrix)
        points_xyz_for_proj = apply_transform(points_xyz_bev, np.linalg.inv(lidar_aug_matrix))

    uv_lidar, depth_lidar = project_lidar_to_image(points_xyz_for_proj, lidar2image)
    proj = np.concatenate([uv_lidar, depth_lidar[:, None]], axis=1)
    proj = (img_aug_matrix[:3, :3] @ proj.T).T + img_aug_matrix[:3, 3][None, :]
    uv_lidar = proj[:, :2]

    mask_lidar = (
        (depth_lidar > 0)
        & (uv_lidar[:, 0] >= 0)
        & (uv_lidar[:, 0] < img_w)
        & (uv_lidar[:, 1] >= 0)
        & (uv_lidar[:, 1] < img_h)
    )

    frustum = create_frustum((img_h, img_w), tuple(args.feature_size), tuple(args.dbound))
    frustum_lidar = frustum_to_lidar(
        frustum,
        cam_intrinsics,
        camera2lidar,
        img_aug_matrix=img_aug_matrix,
        lidar_aug_matrix=lidar_aug_matrix if args.use_augment else None,
    )
    if args.frustum_sample > 1:
        frustum_lidar_vis = frustum_lidar[:: args.frustum_sample]
    else:
        frustum_lidar_vis = frustum_lidar

    uv_frustum, depth_frustum = project_lidar_to_image(frustum_lidar_vis, lidar2image)
    proj_frustum = np.concatenate([uv_frustum, depth_frustum[:, None]], axis=1)
    proj_frustum = (img_aug_matrix[:3, :3] @ proj_frustum.T).T + img_aug_matrix[:3, 3][None, :]
    uv_frustum = proj_frustum[:, :2]
    mask_frustum = (
        (depth_frustum > 0)
        & (uv_frustum[:, 0] >= 0)
        & (uv_frustum[:, 0] < img_w)
        & (uv_frustum[:, 1] >= 0)
        & (uv_frustum[:, 1] < img_h)
    )

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
    ax_bev = fig.add_subplot(gs[:, 0])    # left: BEV spans two rows
    ax_img_lidar = fig.add_subplot(gs[0, 1])  # right-top
    ax_img_frustum = fig.add_subplot(gs[1, 1])  # right-bottom

    ax_img_lidar.imshow(image)
    ax_img_lidar.scatter(
        uv_lidar[mask_lidar, 0],
        uv_lidar[mask_lidar, 1],
        s=1,
        c=depth_lidar[mask_lidar],
        cmap="viridis",
        alpha=0.7,
        label="LiDAR -> image",
    )
    ax_img_lidar.set_title("Image View: LiDAR -> image")
    ax_img_lidar.set_xlim([0, img_w - 1])
    ax_img_lidar.set_ylim([img_h - 1, 0])
    ax_img_lidar.legend(loc="upper right")

    ax_img_frustum.imshow(image)
    ax_img_frustum.scatter(
        uv_frustum[mask_frustum, 0],
        uv_frustum[mask_frustum, 1],
        s=1,
        c="red",
        alpha=0.35,
        label="frustum -> image",
    )
    ax_img_frustum.set_title("Image View: frustum -> image")
    ax_img_frustum.set_xlim([0, img_w - 1])
    ax_img_frustum.set_ylim([img_h - 1, 0])
    ax_img_frustum.legend(loc="upper right")

    ax_bev.scatter(points_xyz_bev[:, 0], points_xyz_bev[:, 1], s=0.5, c="gray", alpha=0.4, label="LiDAR points")
    ax_bev.scatter(
        frustum_lidar_vis[:, 0],
        frustum_lidar_vis[:, 1],
        s=0.3,
        c="red",
        alpha=0.12,
        label="frustum backprojected points",
    )
    ax_bev.set_title("BEV View (LiDAR frame)")
    ax_bev.set_xlabel("x (forward)")
    ax_bev.set_ylabel("y (left)")
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.set_xlim(args.xlim)
    ax_bev.set_ylim(args.ylim)
    ax_bev.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] saved: {output_path}")


"""
python tools/visualize_lidar_image_frustum.py \
  --calib ./datasets/dense_kitti/kitti_image/00/calib.txt \
  --lidar ./datasets/dense_kitti/kitti_odo/00/000123.bin \
  --image ./datasets/dense_kitti/kitti_image/00/image_2/000123.png \
  --output ./proj_debug.png
"""
if __name__ == "__main__":
    main()
