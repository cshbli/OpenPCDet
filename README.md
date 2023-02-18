<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet Quantization Aware Training

## Float model baseline

### Test the trained float model
```
python test.py --cfg_file cfgs/leishen_models/pp_robosense_baseline_test.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth
```

output:
```
Car:90.5452
Truck:90.8873
Bus:79.1617
Non_motor_vehicles:80.0150
Pedestrians:0.0000
```

### Export the float model

```
python export_onnx.py --cfg_file cfgs/leishen_models/pp_robosense_baseline_test.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth
```

The exported onnx model is saved as `leishen_float_model.onnx`.

### Train one more epoch

```
python train.py --cfg_file cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--fix_random_seed --batch_size 4  --epochs 1 \
--pretrained_model ../checkpoints/leishen/checkpoint_epoch_30.pth
```

output:
```
Car:86.2589
Truck:90.3599
Bus:62.8783
Non_motor_vehicles:65.6314
Pedestrians:0.0000
```

## Float model Structure

```
PointPillar(
  (vfe): PillarVFE()
  (backbone_3d): None
  (map_to_bev_module): PointPillarScatter()
  (pfe): None
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU()
      )
      (1): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU()
        (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (14): ReLU()
        (15): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (16): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (17): ReLU()
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
      )
      (1): Sequential(
        (0): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
      )
    )
    (encoder): Sequential(
      (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (4): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (5): ReLU()
    )
    (upsample): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (upsample1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (concat): CatChannel()
    (quant): QuantStub()
  )
  (dense_head): AnchorHeadSingle(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (reg_loss_func): WeightedSmoothL1Loss()
    (dir_loss_func): WeightedCrossEntropyLoss()
    (conv_cls): Conv2d(256, 50, kernel_size=(1, 1), stride=(1, 1))
    (conv_box): Conv2d(256, 70, kernel_size=(1, 1), stride=(1, 1))
    (conv_dir_cls): Conv2d(256, 40, kernel_size=(1, 1), stride=(1, 1))
    (dequant0): DeQuantStub()
    (dequant1): DeQuantStub()
    (dequant2): DeQuantStub()
  )
  (point_head): None
  (roi_head): None
)
```

## Note

* Please make sure to use the latest version of [data_precessor.py](./pcdet/datasets/processor/data_processor.py). It can handle spconv version 1.x and 2.x.

* For pypcd io error

  ```
  File "/home/hongbing/venv/torch1.9.1/lib/python3.8/site-packages/pypcd/pypcd.py", line 15, in <module>
      import io.StringIO as sio
  ModuleNotFoundError: No module named 'io.StringIO'; 'io' is not a package
  ```

  ```
  from io import StringIO as sio
  ```

* Please make sure to use the latest version of `pcdec/ops` for all <b>*.cpp, *.cu and *.h</b>. Otherwise it won't support higher version of PyTorch.  

