from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
# from data_io import *
from skimage.transform import resize
import cv2


# the ETH dataset preprocessed by Yao Yao
class MVSDataset(Dataset):
    def __init__(self, datapath, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath # /datasets/MVS_dataset
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
                num_viewpoint = int(f.readline()) # 330
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scene_name, ref_view, src_views))
                    scene_idx+=1

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
        intrinsic_matrices = []
        extrinsic_matrices = []
        mask_ref = None
        mask_src = []

        # mvs
        # imgs = []
        # mask = None
        # depth = None
        # depth_values = None
        # proj_matrices = []


        for i, id in enumerate(view_ids):
            # NOTE that the id in image file names is 1 indexed
            # print(id)            
            # print(index2prefix_content[id].split()[1])
            # image path
            img_path = os.path.join(scene_path, 'images', index2prefix_content[id].split()[1])

            # depth path
            depth_folder, depth_img = index2prefix_content[id].split()[1].split('/')[0][:-12], index2prefix_content[id].split()[1].split('/')[1].replace('.png','.pfm')
            depth_path = os.path.join(scene_path, 'depths', depth_folder, depth_img)

            # mask path
            mask_folder, mask_img = index2prefix_content[id].split()[1].split('/')[0][:-12], index2prefix_content[id].split()[1].split('/')[1]
            mask_path = os.path.join(scene_path, 'depths', mask_folder, mask_img)
            
            # normal path
            normal_folder, normal_img = index2prefix_content[id].split()[1].split('/')[0][:-12], index2prefix_content[id].split()[1].split('/')[1].replace('.png','.pfm')
            normal_path = os.path.join(scene_path, 'normals', normal_folder, normal_img)

            # camera matrix
            cam_path = os.path.join(scene_path, 'cams', ('{:0>8}_cam.txt').format(id))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(cam_path)
            
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            intrinsic_matrices.append(intrinsics)
            extrinsic_matrices.append(extrinsics)

        # mvs
            imgs.append(self.read_img(img_path))

            if i == 0:  # reference view
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.linspace(depth_min, depth_max, self.ndepths, dtype = np.float32)
                mask = self.read_img(mask_path)
                depth = self.read_depth(depth_path)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth.copy(),
                "depth_values": depth_values,
                "mask": mask.copy()}


if __name__ == "__main__":
    dataset = MVSDataset('MVS_dataset', 'train', 10)
    
    item = dataset[102]
    
    for key in item.keys():
        print(key)
        print(item[key].shape)
    