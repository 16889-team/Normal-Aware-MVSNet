from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from skimage.transform import resize
import cv2


# the ETH dataset
class MVSDataset(Dataset):
    def __init__(self, datapath, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath  # /datasets/MVS_dataset
        # self.model = model
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val"]

        if mode == 'train':
            self.scene_names = ['delivery_area', 'electro', 'forest']
        elif mode == 'val':
            self.scene_names = ['playground', 'terrains']

        self.build_list()

    def build_list(self):
        metas = []
        pair_file = 'cams/pair.txt'
        # index2prefix_file = 'cams/index2prefix.txt'
        scene_idx = 1
        for scene_name in self.scene_names:
            scene_path = os.path.join(self.datapath, 'eth3d', self.mode, scene_name)
            with open(os.path.join(scene_path, pair_file)) as f:
                num_viewpoint = int(f.readline())  # 330
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scene_name, ref_view, src_views))
                    scene_idx += 1

        self.metas = metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        return np.array(img, dtype=np.float32) / 255.

    def read_mask(self, filename):
        img = Image.open(filename)
        mask = np.array(img, dtype=np.bool)
        return mask

    def read_depth(self, filename):
        # read pfm depth file
        return read_pfm(filename)[0]

    def read_normal(self, filename):
        # read normal map
        return read_pfm(filename)[0]

    def transform_view(self, ref_proj_mat, src_proj_mats, depth_src):
        """
        :param ref_proj_mat: size [4, 4]
        :param src_proj_mats: size [N, 4, 4]
        :param depth_src: size [N, 500, 900]
        :return: grid: size[N, 500, 900, 2] range should in (-1, 1)
        """
        N, height, width = depth_src.shape

        XX = np.zeros((N, height, width)).astype(np.float32)
        YY = np.zeros((N, height, width)).astype(np.float32)

        for i in range(N):
            xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
            yy = yy.reshape([-1])
            xx = xx.reshape([-1])
            X = np.vstack((xx, yy, np.ones_like(xx)))  # [500*900, 3]

            D = depth_src[i].reshape((-1))  # [500*900, 1]
            X = np.vstack((X * D, np.ones_like(xx)))  # [N, 500*900, 4]
            X = np.matmul(np.linalg.inv(src_proj_mats[i]), X)
            X = np.matmul(ref_proj_mat, X)
            X /= X[2]
            X = X[:2]

            XX[i] = 2 * ((X[0].reshape([height, width]).astype(np.float32) / (width - 1)) - 0.5)
            YY[i] = 2 * ((X[1].reshape([height, width]).astype(np.float32) / (height - 1)) - 0.5)

        grid = np.stack((XX, YY), -1)
        return grid

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scene_name, ref_view, src_views = meta
        scene_path = os.path.join(self.datapath, 'eth3d', self.mode, scene_name)
        index2prefix_file = open(os.path.join(scene_path, 'cams/index2prefix.txt'))
        index2prefix_content = index2prefix_file.readlines()

        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        # our model
        imgs = []
        imgs_ref = None
        imgs_src = []
        depth_ref = None
        depth_src = []
        normal_ref = None
        normal_src = []
        proj_matrices = []
        intrinsics_inv = None
        all_extrinsics = []
        mask_ref = None
        mask_src = []
        depth_values = None


        for i, id in enumerate(view_ids):
            # NOTE that the id in image file names is 1 indexed
            # image path
            img_path = os.path.join(scene_path, 'images', index2prefix_content[id].split()[1])

            # depth path
            depth_folder, depth_img = index2prefix_content[id].split()[1].split('/')[0][:-12], \
                                      index2prefix_content[id].split()[1].split('/')[1].replace('.png', '.pfm')
            depth_path = os.path.join(scene_path, 'depths', depth_folder, depth_img)

            # mask path
            mask_folder, mask_img = index2prefix_content[id].split()[1].split('/')[0][:-12], \
                                    index2prefix_content[id].split()[1].split('/')[1]
            mask_path = os.path.join(scene_path, 'depths', mask_folder, mask_img)

            # normal path
            normal_folder, normal_img = index2prefix_content[id].split()[1].split('/')[0][:-12], \
                                        index2prefix_content[id].split()[1].split('/')[1].replace('.png', '.pfm')
            normal_path = os.path.join(scene_path, 'normals', normal_folder, normal_img)

            # camera matrix
            cam_path = os.path.join(scene_path, 'cams', ('{:0>8}_cam.txt').format(id))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(cam_path)

            all_extrinsics.append(extrinsics.copy())

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            # our model
            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)[:self.ndepths]
                imgs_ref = self.read_img(img_path)
                depth_ref = self.read_depth(depth_path)
                normal_ref = self.read_normal(normal_path)
                mask_ref = self.read_mask(mask_path)
                intrinsics_inv = np.linalg.inv(intrinsics.copy())
            else:
                imgs_src.append(self.read_img(img_path))
                depth_src.append(self.read_depth(depth_path))
                normal_src.append(self.read_normal(normal_path))
                mask_src.append(self.read_mask(mask_path))

        imgs_src = np.stack(imgs_src)
        depth_src = np.stack(depth_src)
        normal_src = np.stack(normal_src)
        mask_src = np.stack(mask_src)

        proj_matrices = np.stack(proj_matrices)
        extrinsics = np.stack(all_extrinsics)
        # transform view
        grid = self.transform_view(proj_matrices[0, :, :], proj_matrices[1:, :, :], depth_src)

        return {
            "imgs_ref": imgs_ref.copy(),  # (h, w, 3)
            "depth_ref": depth_ref.copy(),  # (h, w,)
            "normal_ref": normal_ref.copy(),  # (h, w, 3)
            "mask_ref": mask_ref.copy(),  # (h, w) bool
            "imgs_src": imgs_src.copy(),  # (N-1, h, w, 3)
            "depth_src": depth_src.copy(),  # (N-1, h, w)
            "normal_src": normal_src.copy(),  # (N-1,h, w, 3)
            "mask_src": mask_src.copy(),  # (N-1, h, w)
            "proj_mat": proj_matrices.copy(),  # (N, 4, 4)
            "intrinsics_inv": intrinsics_inv.copy(),  # (3, 3)
            'extrinsics': extrinsics.copy(),  # (N, 4, 4)
            "depth_values": depth_values.copy(),
            'grid': grid.copy(),  # [N - 1, h, w, 2]
        }


if __name__ == "__main__":
    import torch

    dataset = MVSDataset('MVS_dataset', 'train', 10)

    item = dataset[0]

    for key in item.keys():
        print(key)
        print(item[key].shape)

    depth = item["depth_ref"]
    mask = item["mask_ref"]

    cv2.imshow("0", depth)
    depth[mask] = 0.1
    cv2.waitKey(0)
    cv2.imshow("1", depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dataset.read_mask('MVS_dataset/eth3d/train/delivery_area/depths/images_rig_cam4/1477843917481127523.png')
    print("imgs", item["imgs_ref"].shape)
    print("depth", item["depth_ref"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask_ref"].shape)

    # test homography here
    print(item.keys())
    print("imgs", item["imgs_ref"].shape)
    print("depth", item["depth_ref"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask_ref"].shape)

    ref_img = item["imgs_ref"]
    src_imgs = [item["imgs_src"][i] for i in range(4)]
    ref_proj_mat = item["proj_mat"][0]
    src_proj_mats = [item["proj_mat"][i] for i in range(1, 5)]
    mask = item["mask_src"][0]
    depth = item["depth_src"][0]
    depth_ref = item['depth_ref']

    normal_ref = item['normal_ref']
    normal_src = [item['normal_src'][i] for i in range(4)]

    grid = [item['grid'][i] for i in range(4)]

    e = [item['extrinsics'][i] for i in range(5)]

    e_ref = e[0]
    e_src = e[1]

    R_ref = e_ref[0:3, 0:3]
    R_src = e_src[0:3, 0:3]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    print(height, width)
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])

    X = np.vstack((xx, yy, np.ones_like(xx)))
    print(X[0].max())

    XX = 2 * ((X[0].reshape([height, width]).astype(np.float32) / (width - 1)) - 0.5)
    YY = 2 * ((X[1].reshape([height, width]).astype(np.float32) / (height - 1)) - 0.5)
    print(np.max(YY), np.min(YY))
    print(np.max(XX), np.min(XX))

    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(src_proj_mats[0]), X)
    X = np.matmul(ref_proj_mat, X)
    X /= X[2]
    X = X[:2]

    xx = 2 * ((X[0].reshape([height, width]).astype(np.float32) / (width - 1)) - 0.5)
    yy = 2 * ((X[1].reshape([height, width]).astype(np.float32) / (height - 1)) - 0.5)
    print("---------")
    print(np.max(yy), np.min(yy))
    print(np.max(xx), np.min(xx))
    # import cv2

    print(normal_ref.shape)
    print(R_ref.shape)
    normal_ref = normal_ref.reshape((-1, 3))
    normal_ref = np.matmul(np.linalg.inv(R_ref), normal_ref.T).T
    normal_ref = np.matmul(R_src, normal_ref.T).T
    normal_ref = normal_ref.reshape((500, 900, 3))
    warped = cv2.remap(normal_ref, X[0].reshape([height, width]).astype(np.float32),
                       X[1].reshape([height, width]).astype(np.float32), interpolation=cv2.INTER_LINEAR)

    depth_ref1 = torch.from_numpy(depth_ref.copy()).unsqueeze(0).unsqueeze(3).permute(0, 3, 1, 2)

    grid = grid[0]

    grid = torch.from_numpy(grid).unsqueeze(0)
    import torch.nn.functional as F

    warped2 = F.grid_sample(depth_ref1, grid, align_corners=False).squeeze(0).permute(1, 2, 0).numpy()

    warped2[mask[:, :] == 0] = 0

    warped[mask[:, :] == 0] = 0

    cv2.imwrite('../tmp0.png', depth_ref[:, :, ::-1] * 255)
    cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    cv2.imwrite('../tmp2.png', depth[:, :] * 10)
    cv2.imwrite('../tmp3.png', warped2[:, :] * 10)



