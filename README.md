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

### QAT model baseline

### Test the QAT model

```
python test_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--ckpt_qat ../checkpoints/leishen/qat/checkpoint_epoch_25.pth \
--cwd ~/Projects/OpenPCDet/tools --batch_size 6 \
--output_dir ../output/leishen_models/qat_baseline
```

output:
```
Car:82.9962
Truck:71.6373
Bus:33.1491
Non_motor_vehicles:51.7760
Pedestrians:0.0000
```

### Train one epoch

```
python train_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--pretrained_model ../checkpoints/leishen/qat/checkpoint_epoch_25.pth \
--cwd /home/hongbing/Projects/OpenPCDet/tools \
--batch_size 4 --output_dir ../output_minmax --epochs 1
```
output: 

```
Car:50.4889
Truck:40.8658
Bus:0.4545
Non_motor_vehicles:14.6391
Pedestrians:0.0000
```

### Train one epoch without activation quantization

```
python train_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--pretrained_model ../checkpoints/leishen/qat/checkpoint_epoch_25.pth \
--cwd /home/hongbing/Projects/OpenPCDet/tools \
--batch_size 4 --output_dir ../output_minmax --epochs 1
```
output: 

```
Car:68.3408
Truck:34.3958
Bus:9.0909
Non_motor_vehicles:22.5743
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

## Quantized Model Structure

```
PointPillar(
  (vfe): PillarVFE()
  (backbone_3d): None
  (map_to_bev_module): PointPillarScatter()
  (pfe): None
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(
          128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (3): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (6): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (9): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
      )
      (1): Sequential(
        (0): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (3): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (6): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (9): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (12): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (13): Identity()
        (14): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (15): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (16): Identity()
        (17): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(
          64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (1): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
      )
      (1): Sequential(
        (0): ConvTranspose2d(
          128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (1): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
            (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
          )
        )
      )
    )
    (encoder): Sequential(
      (0): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
      )
      (1): Identity()
      (2): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
      )
      (3): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
      )
      (4): Identity()
      (5): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
          (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
        )
      )
    )
    (upsample): ConvTranspose2d(
      128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (upsample1): Conv2d(
      64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (cat1): CatChannel(
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (quant): QuantStub(
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
  )
  (dense_head): AnchorHeadSingle(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (reg_loss_func): WeightedSmoothL1Loss()
    (dir_loss_func): WeightedCrossEntropyLoss()
    (conv_cls): Conv2d(
      256, 50, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (conv_box): Conv2d(
      256, 70, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (conv_dir_cls): Conv2d(
      256, 40, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (dequant0): DeQuantStub()
    (dequant1): DeQuantStub()
    (dequant2): DeQuantStub()
  )
  (point_head): None
  (roi_head): None
)
```

## Example Quantized Model with Parameters

```
PointPillar(
  (vfe): PillarVFE()
  (backbone_3d): None
  (map_to_bev_module): PointPillarScatter()
  (pfe): None
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(
          128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.001953125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.1304902881383896, max_val=0.09414418786764145)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=2.0, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-128.0633087158203, max_val=76.6624755859375)
          )
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=76.0)
          )
        )
        (3): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.3085933029651642, max_val=0.23773878812789917)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-94.27420806884766, max_val=54.66197967529297)
          )
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=55.0)
          )
        )
        (6): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.3398781418800354, max_val=0.34247320890426636)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-64.14228057861328, max_val=61.98301315307617)
          )
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=62.0)
          )
        )
        (9): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.001953125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.23828257620334625, max_val=0.23765724897384644)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-60.65795135498047, max_val=40.39590072631836)
          )
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=40.5)
          )
        )
      )
      (1): Sequential(
        (0): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.23526152968406677, max_val=0.2521400451660156)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-50.09825134277344, max_val=38.744598388671875)
          )
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=38.5)
          )
        )
        (3): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.001953125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.2121623307466507, max_val=0.17973963916301727)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-35.682552337646484, max_val=19.165653228759766)
          )
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=19.0)
          )
        )
        (6): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.32826364040374756, max_val=0.2478192299604416)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-21.25888442993164, max_val=14.671653747558594)
          )
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=14.75)
          )
        )
        (9): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.32638415694236755, max_val=0.26751452684402466)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-12.750164985656738, max_val=9.106963157653809)
          )
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=9.125)
          )
        )
        (12): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.36336272954940796, max_val=0.3597654402256012)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-9.826794624328613, max_val=6.534886360168457)
          )
        )
        (13): Identity()
        (14): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.5)
          )
        )
        (15): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.34332728385925293, max_val=0.29694288969039917)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-8.270151138305664, max_val=6.023543834686279)
          )
        )
        (16): Identity()
        (17): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.0)
          )
        )
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(
          64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0078125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=1.0)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=40.18359375)
          )
        )
        (1): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0009765625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.07276688516139984, max_val=0.055559203028678894)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-19.193119049072266, max_val=3.9230291843414307)
          )
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=4.0)
          )
        )
      )
      (1): Sequential(
        (0): ConvTranspose2d(
          128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0078125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=1.0)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=5.953125)
          )
        )
        (1): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0009765625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.08716867864131927, max_val=0.08934550732374191)
          )
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-4.760479927062988, max_val=3.493166446685791)
          )
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=4.0)
          )
        )
      )
    )
    (encoder): Sequential(
      (0): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-13.345190048217773, max_val=11.10741138458252)
        )
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-85.2032470703125, max_val=50.28233337402344)
        )
      )
      (1): Identity()
      (2): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=50.0)
        )
      )
      (3): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.846185326576233, max_val=3.8979461193084717)
        )
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-98.28540802001953, max_val=127.10458374023438)
        )
      )
      (4): Identity()
      (5): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=1.0, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=127.0)
        )
      )
    )
    (upsample): ConvTranspose2d(
      128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (upsample1): Conv2d(
      64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=tensor([1.], device='cuda:0'), zero_point=tensor([0], device='cuda:0')
        (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
      )
    )
    (cat1): CatChannel(
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=4.0)
      )
    )
    (quant): QuantStub(
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-3.499997615814209, max_val=3.499980926513672)
      )
    )
  )
  (dense_head): AnchorHeadSingle(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (reg_loss_func): WeightedSmoothL1Loss()
    (dir_loss_func): WeightedCrossEntropyLoss()
    (conv_cls): Conv2d(
      256, 50, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.648536682128906, max_val=1.8285142183303833)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-51.46748733520508, max_val=3.9925436973571777)
      )
    )
    (conv_box): Conv2d(
      256, 70, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0078125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.8128319382667542, max_val=0.745008111000061)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-2.7310738563537598, max_val=1.7538713216781616)
      )
    )
    (conv_dir_cls): Conv2d(
      256, 40, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-5.742468357086182, max_val=3.3943777084350586)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-34.161468505859375, max_val=18.736011505126953)
      )
    )
    (dequant0): DeQuantStub()
    (dequant1): DeQuantStub()
    (dequant2): DeQuantStub()
  )
  (point_head): None
  (roi_head): None
)
```

```
{'Version': '0.2', 'Graph': OrderedDict([('activation_quant_mode', 'per_tensor'), ('bias_quant_mode', 'dynamic'), ('quant_data_type', 'int8'), ('quant_max', 127), ('quant_min', -128)]), 'Tensor': OrderedDict([('backbone_2d.blocks.0.0.weight', OrderedDict([('quant_scale', 0.001953125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.0.bias', OrderedDict([('range_min', -2.2350246906280518), ('range_max', 1.773344874382019), ('quant_scale', 1.862645149230957e-09), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.3.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.3.bias', OrderedDict([('range_min', -2.2804548740386963), ('range_max', 1.9319746494293213), ('quant_scale', 1.862645149230957e-09), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.6.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.6.bias', OrderedDict([('range_min', -1.2016290426254272), ('range_max', 1.485540509223938), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.9.weight', OrderedDict([('quant_scale', 0.001953125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.0.9.bias', OrderedDict([('range_min', -1.398720622062683), ('range_max', 1.4279413223266602), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.0.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.0.bias', OrderedDict([('range_min', -1.7888931035995483), ('range_max', 1.909980058670044), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.3.weight', OrderedDict([('quant_scale', 0.001953125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.3.bias', OrderedDict([('range_min', -0.5948971509933472), ('range_max', 1.8151755332946777), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.6.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.6.bias', OrderedDict([('range_min', -0.9825441241264343), ('range_max', 1.5164093971252441), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.9.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.9.bias', OrderedDict([('range_min', -0.730673611164093), ('range_max', 1.2469384670257568), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.12.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.12.bias', OrderedDict([('range_min', -0.6871170401573181), ('range_max', 0.9071369767189026), ('quant_scale', 4.656612873077393e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.15.weight', OrderedDict([('quant_scale', 0.00390625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.blocks.1.15.bias', OrderedDict([('range_min', -1.0189499855041504), ('range_max', 1.9159348011016846), ('quant_scale', 9.313225746154785e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.0.0.weight', OrderedDict([('quant_scale', 0.0078125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.0.1.weight', OrderedDict([('quant_scale', 0.0009765625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.0.1.bias', OrderedDict([('range_min', -0.44450825452804565), ('range_max', 0.3844316601753235), ('quant_scale', 2.3283064365386963e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.1.0.weight', OrderedDict([('quant_scale', 0.0078125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.1.0.bias', OrderedDict([('range_min', 0.0), ('range_max', 0.0), ('quant_scale', 4.656612873077393e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.1.1.weight', OrderedDict([('quant_scale', 0.0009765625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.deblocks.1.1.bias', OrderedDict([('range_min', -0.6573569774627686), ('range_max', 0.8397030234336853), ('quant_scale', 4.656612873077393e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.encoder.0.weight', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.encoder.0.bias', OrderedDict([('range_min', -0.5732949376106262), ('range_max', 0.6557602286338806), ('quant_scale', 4.656612873077393e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.encoder.3.weight', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.encoder.3.bias', OrderedDict([('range_min', -6.195773124694824), ('range_max', 8.675138473510742), ('quant_scale', 7.450580596923828e-09), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.upsample.weight', OrderedDict([('range_min', inf), ('range_max', -inf), ('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.upsample.bias', OrderedDict([('range_min', 0.0), ('range_max', 0.0), ('quant_scale', 4.656612873077393e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('backbone_2d.upsample1.weight', OrderedDict([('range_min', inf), ('range_max', -inf), ('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('dense_head.conv_cls.weight', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('dense_head.conv_cls.bias', OrderedDict([('range_min', -5.077279090881348), ('range_max', -2.798128843307495), ('quant_scale', 3.725290298461914e-09), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('dense_head.conv_box.weight', OrderedDict([('quant_scale', 0.0078125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('dense_head.conv_box.bias', OrderedDict([('range_min', -0.41760581731796265), ('range_max', 0.05137377232313156), ('quant_scale', 2.3283064365386963e-10), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), ('dense_head.conv_dir_cls.weight', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('dense_head.conv_dir_cls.bias', OrderedDict([('range_min', -3.2116434574127197), ('range_max', 1.5283961296081543), ('quant_scale', 1.862645149230957e-09), ('quant_zero_point', 0.0), ('quant_min', -2147483648), ('quant_max', 2147483647), ('quant_type', 0), ('quant_data_type', 'int32'), ('rounding_scheme', 'round_to_even')])), 
('X.1', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('452', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('459', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('472', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('479', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 

('492', OrderedDict([('quant_scale', 2.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('499', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 

('512', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('519', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 

('532', OrderedDict([('quant_scale', 1.0), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])),
('539', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])),

('552', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 
('559', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), 

('572', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('585', OrderedDict([('quant_scale', 0.25), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('592', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('605', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('612', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('625', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('632', OrderedDict([('quant_scale', 0.25), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('645', OrderedDict([('quant_scale', 0.25), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('652', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('665', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('672', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('685', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('692', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('705', OrderedDict([('quant_scale', 0.125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('712', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('725', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('738', OrderedDict([('quant_scale', 0.0625), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('745', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('752', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('765', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('778', OrderedDict([('quant_scale', 0.03125), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')])), ('791', OrderedDict([('quant_scale', 0.5), ('quant_zero_point', 0.0), ('quant_min', -128), ('quant_max', 127), ('quant_type', 0), ('quant_data_type', 'int8'), ('rounding_scheme', 'round_to_even')]))])}

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

