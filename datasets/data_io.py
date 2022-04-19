from PIL import Image
from tqdm import tqdm
import os
import imageio
import numpy as np
import re
import sys
import glob
import cv2

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def create_mask(filename, outpath):
    '''
    create mask for depth map (.pfm file)
    '''
    data, _  = read_pfm(filename)
    mask = np.zeros_like(data, dtype = np.uint8)
    # mask = np.zeros_like(data)

    mask[data>0] = 255
    # mask = mask.astype(np.uint8)

    imageio.imwrite(outpath, mask)

    return

def resize_img(filepath):
    im = Image.open(filepath)
    # im = cv2.imread(filepath)
    # print(im.shape)
    if im.size != (900, 500):
        im = im.resize((900, 500))
        im.save(filepath)

def resize_depth(filepath):
    '''
    in-place resize depth images
    '''
    data, _  = read_pfm(filepath)
    cv2.imwrite('depth.png', data)
    dim = (900, 500)
    if data.size != (900, 500):
        resized = cv2.resize(data, dim, interpolation = cv2.INTER_AREA)
        resized_arr = np.array(resized)
        save_pfm(filepath, resized_arr, scale=1)

def test():
    img = np.zeros([500,500,3])

    img[:,:,0] = np.ones([500,500])*64/255.0
    img[:,:,1] = np.ones([500,500])*128/255.0
    img[:,:,2] = np.ones([500,500])*192/255.0

    cv2.imwrite('color_img.jpg', img)
    cv2.imshow("image", img)
    cv2.waitKey()

def visualize_depth(filepath, colormap=True):
    data, _  = read_pfm(filepath)
    data /= data.max()
    data*=255
    data = data.astype(np.uint8)
    
    if colormap:
        colored = cv2.applyColorMap(data, cv2.COLORMAP_JET)

    cv2.imshow("Display window", colored)
    k = cv2.waitKey(0)

def visualize_normal(filepath):
    normal, _  = read_pfm(filepath)
    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    # cv2.imwrite("normal0.png", normal[:, :, ::-1])
    normal = normal.astype(np.uint8)
    cv2.imshow("Display window", normal)
    k = cv2.waitKey(0)

def main():
    # convert all images and depth map to 500 x 900
    path_to_eth3d = os.path.join('MVS_dataset', 'eth3d')
    path_to_sets = glob.glob(os.path.join(path_to_eth3d, '*')) # train, test, val
    for path_to_set in path_to_sets:
        print('processing: ', path_to_set)
        if path_to_set == 'MVS_dataset/eth3d/test':
            # in test set, there's only images
            path_to_scenes = glob.glob(os.path.join(path_to_set, '*')) # delivery_area, etc
            print(path_to_scenes)
            for path_to_scene in path_to_scenes:
                folders = os.listdir(path_to_scene)
                for folder in folders:
                    if folder == 'images':
                        path_to_images = glob.glob(os.path.join(path_to_scene, folder, '*.jpg'))
                        for i, path_to_image in enumerate(tqdm(path_to_images)):
                            resize_img(path_to_image)
        else:
            path_to_scenes = glob.glob(os.path.join(path_to_set, '*')) # delivery_area, etc
            for path_to_scene in path_to_scenes:
                folders = os.listdir(path_to_scene)
                if 'normals' not in folders:
                    os.mkdir(os.path.join(path_to_scene, 'normals'))
                    sub_folders = os.listdir(os.path.join(path_to_scene, 'depths'))
                    for sub_folder in sub_folders:
                        os.mkdir(os.path.join(path_to_scene, 'normals', sub_folder))
                # return
                for folder in folders:
                    if folder == 'depths':
                        print('Processing depth images in ', path_to_scene)
                        sub_folders = glob.glob(os.path.join(path_to_scene, folder, '*'))
                        # sub_folders = os.listdir(os.path.join(path_to_scene, folder)) # images_rig_cam4 etc.
                        for sub_folder in sub_folders:
                            path_to_images = glob.glob(os.path.join(sub_folder, '*.pfm'))
                            for i, path_to_image in enumerate(tqdm(path_to_images)):
                                # resize depth image
                                resize_depth(path_to_image)
                                # create mask
                                if path_to_image.replace('.pfm', '.png') not in path_to_images:
                                    create_mask(path_to_image, path_to_image.replace('.pfm', '.png'))
                                # create normal map
                                create_normal(path_to_image, path_to_image.replace('depths','normals'))

                    if folder == 'images':
                        print('Processing images in ', path_to_scene)
                        sub_folders = glob.glob(os.path.join(path_to_scene, folder, '*'))
                        for sub_folder in sub_folders:
                            path_to_images = glob.glob(os.path.join(sub_folder, '*.png'))
                            for i, path_to_image in enumerate(tqdm(path_to_images)):
                                resize_img(path_to_image)

def create_normal(filepath, outpath):
    '''
    Attention: normal map created is a vector, though normalized, its range is in [-1, 1]
    '''
    depth, _  = read_pfm(filepath)
    depth *= 1000
    depth_resized = cv2.resize(depth, (900, 500))
    # depth = cv2.GaussianBlur(depth_resized, [3,3], cv2.BORDER_DEFAULT) # not helping
    zy, zx = np.gradient(depth)  
    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    # mask
    normal[depth==0, 0] = 0
    normal[depth==0, 1] = 0
    normal[depth==0, 2] = 0
    # normal = cv2.GaussianBlur(normal, [3,3], cv2.BORDER_DEFAULT) # not helping
    save_pfm(outpath, normal)

    return 

if __name__ == "__main__":

    main()

    # visualize_depth('MVS_dataset/eth3d/train/delivery_area/depths/images_rig_cam4/1477843917481127523.pfm')
    # create_normal('MVS_dataset/eth3d/train/forest/depths/images_rig_cam4/1474982922066603972.pfm', 'test.pfm')
    # create_normal('MVS_dataset/eth3d/train/delivery_area/depths/images_rig_cam4/1477843917481127523.pfm', 'test2.pfm')

    # visualize_normal('MVS_dataset/1477843917481127523.pfm')
    # visualize_normal('')
    # /Users/chenhaoyang/tmp/Normal-Aware-MVSNet/datasets/eth3d/playground/normals/images_rig_cam4_undistorted/1477833684658155598.pfm