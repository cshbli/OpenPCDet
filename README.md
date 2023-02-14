<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet Quantization Aware Training

## Changelog
[2023-02-14] 
  * Disable evaluation output of bbox, bev and aos and AP_R40
  * Only output evaluation result of the first `min_overlaps`
  * Disable resuming from checkpoint if `pretrained_model` is specified

## Introduction

### Test

```
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 16 --ckpt ../checkpoints/pointpillar_7728.pth
```

Baseline result:
```
Car AP@0.70, 0.70, 0.70:
3d   AP:86.4617, 77.2839, 74.6530
Pedestrian AP@0.50, 0.50, 0.50:
3d   AP:57.7500, 52.2916, 47.9072
Cyclist AP@0.50, 0.50, 0.50:
3d   AP:80.0568, 62.6873, 59.7069
```

### Train

```
python train.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 8  --epochs 1 --pretrained_model ../checkpoints/pointpillar_7728.pth --fix_random_seed 
```

