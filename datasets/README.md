# Dataset

Pre-processed ETH3D dataset available at https://cmu.box.com/s/mk4w3tspxrn49r2fzbr3x98pi3g2v60f

Download and unzip at `datasets`, the path would be `datasets/MVS_dataset/eth3d/....`

# How to generate ground truth normal maps (make sure you have downloaded the prepossed dataset and installed open3d)

In this dir, run 

```
!python normal_generation.py --nviews 3/5/7 --datapath eth3d//delivery_area or electro or forest or playground or terrains
```
