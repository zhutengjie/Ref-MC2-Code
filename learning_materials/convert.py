import pdb
import cv2
import json
import torch
import trimesh
import numpy as np
from pathlib import Path

# https://raw.githubusercontent.com/yenchenlin/nerf-pytorch/master/load_blender.py
from load_blender import load_blender_data
np.set_printoptions(suppress=True)


def main():
    case = 'table_horse'
    data_root = Path('/data/cz/nvdiffrecmc-mine/data/nerf_synthetic')
    data_dir = data_root / case

    images_all, poses_all, render_poses, hwf, i_split = load_blender_data(data_dir, half_res=False)
    images_all[...,:3] = images_all[...,:3]*images_all[...,-1:] + (1.-images_all[...,-1:])  # to white bg
    images_all = images_all.astype(np.float32)
    # images_all: (400, 800, 800, 4)  float32 范围[0, 1]
    # poses_all: (400, 4, 4)  column-major，求inv才是R_t_gl
    # render_poses: (40, 4, 4)
    # hwf: h=w=800, f=1111
    # i_split: len=3的list  分别表示train/val/test在400中的索引

    flag = 0
    if flag:
        ply_path = data_root / f'extracted_ply/{case}.ply'
        mesh = trimesh.load(ply_path)
        verts_origin = np.array(mesh.vertices, dtype=np.float32)
        tris = np.array(mesh.faces, dtype=np.int32)
        print(verts_origin.shape, tris.shape)
        dists = np.sqrt((verts_origin ** 2).sum(axis=1))
        max_dist = dists.max()

    else:
        M = {
            'chair': 1.310619, 'drums': 1.250301,
            'ficus': 1.2513705, 'hotdog': 1.2886027,
            'lego': 1.3557879, 'materials': 1.4068246,
            'mic': 1.4869413, 'ship': 1.4760181,
            'toaster_materials_rough': 1.4068246,
            'table_horse':1.4068246
        }
        max_dist = M[case]


    std_radius = 1.0 * 0.98   # NeuS标准空间下的最大半径尺寸
    scale = std_radius / max_dist
    print(f'scale={scale}')

    if flag:
        print('\n【origin】')
        print_verts(verts_origin)
        
        verts_origin *= scale
        print('【to NeuS】')
        print_verts(verts_origin)


    # scale_mat_inv的意义是，从物体原始空间转换到NeuS标准空间
    scale_mat_inv = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
    scale_mat = np.linalg.inv(scale_mat_inv)

    img_h, img_w, focal = hwf
    K_img = np.identity(4, dtype=np.float32)
    K_img[0, 0] = focal
    K_img[1, 1] = focal
    K_img[0, 2] = img_w / 2
    K_img[1, 2] = img_h / 2
    

    M = {}
    output_dir = Path('public_data') / case

    for k, index in enumerate(i_split[0]):
        if k % 20 == 0:
            print('k =', k)

        M[f'scale_mat_{k}'] = scale_mat
        M[f'scale_mat_inv_{k}'] = scale_mat_inv

        c2w_gl = poses_all[index]
        R_t_gl = np.linalg.inv(c2w_gl)  # column-major格式
        R_t_cv = np.diag([1, -1, -1, 1]).astype(np.float32) @ R_t_gl
        assert R_t_cv[2, 3] > 0

        world_mat = K_img @ R_t_cv      # column-major，左乘
        M[f'world_mat_{k}'] = world_mat

        img_rgb = images_all[index, :, :, :3]
        img_name = 'image/' + ('%03d.png' % k)
        output_path = output_dir / img_name
        save_float32_rgb(output_path, img_rgb)

        mask = images_all[index, :, :, 3]
        output_path = output_dir / 'mask' / ('%03d.png' % k)
        save_float32_gray(output_path, mask)

    output_path = output_dir / 'cameras_sphere.npz'
    np.savez(output_path, **M)
    print('【Done】')



def save_float32_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def save_float32_gray(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def print_verts(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    assert x.shape[1] == 3
    print('shape =', x.shape)
    for c in range(3):
        min_ = x[:, c].min()
        max_ = x[:, c].max()
        mid = (max_ + min_) / 2
        span = max_ - min_
        print('c=%d, min_max=[%f, %f], mid=%f, span=%f' % (c, min_, max_, mid, span))
    dist = np.linalg.norm(x, axis=1)
    print('max_dist=%f' % dist.max())
    print()


if __name__ == '__main__':
    main()