<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet ONNX model evaluation

## Float model baseline

### Test the trained float model

Please note the onnx model path is specified in the `pp_robosense_baseline_onnx.yaml` as:

The "leishen_float_model.onnx" was exported from float model "checkpoint_epoch_30.pth".

```
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5]
        LAYER_STRIDES: [2, 2]
        NUM_FILTERS: [64, 128]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [128, 128]
        USE_ONNX: True
        PATH: /home/hongbing/Projects/OpenPCDet/checkpoints/leishen/leishen_float_model.onnx        
        INPUT: inputs        
        # PATH: /home/hongbing/Projects/OpenPCDet/checkpoints/leishen/qat/quant_model.onnx
        # INPUT: X.1
```

```
python test.py --cfg_file cfgs/leishen_models/pp_robosense_baseline_onnx.yaml --batch_size 1 --ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth
```

output is the same as PyTorch model:
```
4950/4950 [01:23<00:00, 59.53it/s, recall_0.3=(0, 47612) / 53122]
2023-02-20 05:15:26,666   INFO  *************** Performance of EPOCH 30 *****************
2023-02-20 05:15:26,667   INFO  Generate label finished(sec_per_example: 0.0168 second).
2023-02-20 05:15:26,667   INFO  recall_roi_0.3: 0.000000
2023-02-20 05:15:26,667   INFO  recall_rcnn_0.3: 0.896276
2023-02-20 05:15:26,667   INFO  recall_roi_0.5: 0.000000
2023-02-20 05:15:26,667   INFO  recall_rcnn_0.5: 0.850589
2023-02-20 05:15:26,667   INFO  recall_roi_0.7: 0.000000
2023-02-20 05:15:26,667   INFO  recall_rcnn_0.7: 0.694759
2023-02-20 05:15:26,675   INFO  Average predicted number of objects(4950 samples): 18.749

Car:90.5452
Truck:90.8873
Bus:79.1617
Non_motor_vehicles:80.0150
Pedestrians:0.0000
```

## QAT per-channel model with 60 samples

### Test the BSTNNX ONNX model

Please note the onnx model path is specified in the `pp_robosense_baseline_onnx.yaml` as:

The "quant_model.onnx" was exported from QAT model "checkpoint_epoch_25.pth".

```
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5]
        LAYER_STRIDES: [2, 2]
        NUM_FILTERS: [64, 128]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [128, 128]
        USE_ONNX: True
        PATH: /home/hongbing/Projects/OpenPCDet_onnx/output/leishen_models/qat_per_channel/quant_model.onnx        
        INPUT: X.1
```

```
python test.py --cfg_file cfgs/leishen_models/pp_robosense_baseline_onnx.yaml --batch_size 1 --ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth
```

output:
```
60/60 [01:28<00:00,  1.47s/it, recall_0.3=(0, 526) / 721]
2023-03-09 20:19:01,570   INFO  *************** Performance of EPOCH 30 *****************
2023-03-09 20:19:01,570   INFO  Generate label finished(sec_per_example: 1.4693 second).
2023-03-09 20:19:01,570   INFO  recall_roi_0.3: 0.000000
2023-03-09 20:19:01,570   INFO  recall_rcnn_0.3: 0.729542
2023-03-09 20:19:01,570   INFO  recall_roi_0.5: 0.000000
2023-03-09 20:19:01,570   INFO  recall_rcnn_0.5: 0.542302
2023-03-09 20:19:01,570   INFO  recall_roi_0.7: 0.000000
2023-03-09 20:19:01,570   INFO  recall_rcnn_0.7: 0.184466
2023-03-09 20:19:01,570   INFO  Average predicted number of objects(60 samples): 95.033

Car:71.9900
Truck:69.9686
Bus:0.0000
Non_motor_vehicles:7.6891
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

* bstnnx_training: `fuse_model` function should use `inplace=False`, otherwise it may crash on some computers.

    ```
    if non_empty(modules_to_fuse):
        model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False,
            fuse_custom_config_dict=fuse_custom_config_dict, **kwargs)
    ```

