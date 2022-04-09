import os
import numpy as np
from data_io import read_pfm
import cv2
import open3d as o3d
from sklearn.preprocessing import normalize
import argparse


class NormalGeneration():
    def __init__(self, datapath, nviews):
        self.datapath = datapath
        self.nviews = nviews
        self.interval_scale = 1.06
        self.index2prefix = {}
        self.num_viewpoint = 0
        self.size = [500, 900]

    def read_index2prefix(self, path):
        with open(path) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        self.num_viewpoint = int(lines[0])
        for pair in lines[1:]:
            index, path = pair.split()
            self.index2prefix[int(index)] = path

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

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_resized = cv2.resize(depth, (self.size[1], self.size[0]))
        return depth_resized

    def create_pairs(self):
        pair_path = os.path.join(self.datapath, 'cams/pair.txt')
        pairs = []
        with open(pair_path) as f:
            num_viewpoint = int(f.readline())
            assert num_viewpoint == self.num_viewpoint
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                pairs.append((ref_view, src_views))
        return pairs

    def read_cam_depth(self, index):
        cam_info_path = os.path.join(self.datapath, os.path.join('cams', '{:0>8}_cam.txt'.format(index)))
        cam_info = self.read_cam_file(cam_info_path)
        cam_path, depth_idx_path = str.split(os.path.splitext(self.index2prefix[index])[0], '/')
        depth_path = os.path.join(self.datapath,
                                  os.path.join('depths', os.path.join(cam_path[:15], "{}.pfm".format(depth_idx_path))))
        depth = self.read_depth(depth_path)
        return cam_info, depth

    def generate_pcd(self, pairs):
        all_pcd = o3d.geometry.PointCloud()
        for pair in pairs:
            cam_info, depth = pair
            cv2.imwrite('depth.png', depth)

            depth_o3d = o3d.geometry.Image(depth)
            intrinsics, extrinsics, _, _ = cam_info
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=self.size[0], height=self.size[1],
                                                               fx=intrinsics[0][0], fy=intrinsics[1][1],
                                                               cx=intrinsics[0][2], cy=intrinsics[1][2])
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, o3d_intrinsics, extrinsics,
                                                                  project_valid_depth_only=False)
            all_pcd += pcd
        return all_pcd

    def generate_normal(self, pcd):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=15))
        pcd.normalize_normals()
        normal = np.asarray(pcd.normals)[:self.size[0] * self.size[1], :].reshape((self.size[0], self.size[1], 3))
        return normal

    def generation(self):
        self.read_index2prefix(os.path.join(self.datapath, os.path.join('cams', 'index2prefix.txt')))
        pairs = self.create_pairs()
        for i in range(len(pairs)):
            print("Generate #{} normal".format(i))
            pair = pairs[i]
            ref_view, src_views = pair
            all_pair = []
            ref_cam, ref_depth = self.read_cam_depth(ref_view)
            all_pair.append([ref_cam, ref_depth])
            for j in range(self.nviews):
                src_view = src_views[j]
                src_cam, src_depth = self.read_cam_depth(src_view)
                all_pair.append([src_cam, src_depth])
            pcd = self.generate_pcd(all_pair)
            normal = self.generate_normal(pcd)
            cam_path, idx_path = str.split(os.path.splitext(self.index2prefix[ref_view])[0], '/')
            normals_path = os.path.join(self.datapath, 'normals')
            normal_idx_path = os.path.join(normals_path, os.path.join(cam_path, '{}.png'.format(idx_path)))
            if not os.path.exists(normals_path):
                os.makedirs(normals_path)
            if not os.path.exists(os.path.join(normals_path, cam_path)):
                os.makedirs(os.path.join(normals_path, cam_path))
            cv2.imwrite(normal_idx_path, normal)
        print("Generation is done")
        return


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='eth3d\\delivery_area')
    parser.add_argument('--nviews', type=int, default=3)
    return parser


def main(args):
    datapath = args.datapath
    nview = args.nviews
    ng = NormalGeneration(datapath, nview)
    ng.generation()
    print("Generate for {} is done".format(datapath))
    return


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
