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

### Test the QAT model by disable quantizations after Conv

```
test_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth \ --ckpt_qat /barn4/jishengchen/for_hongbing/per_tensor_minmax_no_quantile/checkpoint_epoch_25.pth \
--batch_size 24 --output_dir ../output_no_conv_quant
```

output:

```
Car:43.3007
Truck:37.1900
Bus:3.6364
Non_motor_vehicles:23.8711
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

* Evaluate the exported ONNX model inside `leishen_onnx` branch

```
python test.py --cfg_file ../tools/cfgs/leishen_models/pp_robosense_baseline_onnx_per_tensor.yaml --batch_size 1 --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth
```

output: 

```
Car:68.3894
Truck:34.9785
Bus:9.0909
Non_motor_vehicles:22.5661
Pedestrians:0.0000
```

* The acurracy of the exported ONNX model is very close to the QAT PyTorch model.

### Train one epoch without activation quantization starting from float model

```
python train_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--batch_size 4 --output_dir ../output/leishen_models/qat_no_activation_quant_from_float --epochs 1 
```

output:

```
Car:65.3343
Truck:56.7995
Bus:3.6335
Non_motor_vehicles:23.4490
Pedestrians:0.0000
```

* Compared to the loading from pretrained QAT model, there have not too much differences.

## Train without Conv Quant

### UINT8 after ReLU

```
python train_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth \
--batch_size 24 --output_dir ../output_hongbing_no_conv_quant --epochs 25
```

Training log:

```
Epoch 0 Average Loss=8.8575
Epoch 1 Average Loss=7.9371
Epoch 2 Average Loss=7.3167
Epoch 3 Average Loss=6.8873
Epoch 4 Average Loss=6.8032
Epoch 5 Average Loss=6.3705
Epoch 6 Average Loss=6.0901
Epoch 7 Average Loss=6.0290
Epoch 8 Average Loss=5.7352
Epoch 9 Average Loss=5.7318
Epoch 10 Average Loss=5.2275
Epoch 11 Average Loss=5.0312
Epoch 12 Average Loss=4.7033
Epoch 13 Average Loss=4.4788
Epoch 14 Average Loss=4.2745
Epoch 15 Average Loss=3.8741
Epoch 16 Average Loss=3.9344
Epoch 17 Average Loss=3.8669
Epoch 18 Average Loss=3.7537
Epoch 19 Average Loss=3.4595
Epoch 20 Average Loss=3.7056
Epoch 21 Average Loss=3.6252
Epoch 22 Average Loss=3.6571
Epoch 23 Average Loss=3.4010
Epoch 24 Average Loss=3.4123
Training is Done

Start evaluation...
2023-02-21 05:12:16,779   INFO  ********************** Evaluation results  **********************

2023-02-21 05:12:16,781   INFO  Car:70.0317
Truck:49.2403
Bus:3.4887
Non_motor_vehicles:20.2594
Pedestrians:0.0000
```

### checkpoint 25

```
python test_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --ckpt_qat ../output_hongbing_no_conv_quant/ckpt/checkpoint_epoch_25.pth --batch_size 24 --output_dir ../output_no_conv_quant
```

```
Car:70.0317
Truck:49.2403
Bus:3.4887
Non_motor_vehicles:20.2594
Pedestrians:0.0000
```

### checkpoint 1

```
python test_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --ckpt_qat ../output_hongbing_no_conv_quant/ckpt/checkpoint_epoch_1.pth --batch_size 24 --output_dir ../output_no_conv_quant
```

```
Car:76.7208
Truck:71.2916
Bus:15.8447
Non_motor_vehicles:51.0582
Pedestrians:0.0000
```

### checkpoint 2

```
python test_qat_xinrui.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --ckpt_qat ../output_hongbing_no_conv_quant/ckpt/checkpoint_epoch_2.pth --batch_size 24 --output_dir ../output_no_conv_quant
```

```
Car:71.2157
Truck:65.7383
Bus:16.2125
Non_motor_vehicles:40.0628
Pedestrians:0.0000
```

BSTNNX 300 stage ONNX model

```
Car:76.4099
Truck:70.9944
Bus:17.5325
Non_motor_vehicles:49.3936
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

## Quantized Model Structure with default QAT flow

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

* While using auto_detect fuse_modules, please use batch_size 1 for the random input sample, otherwise there may have some out of memory process `killed` error. 
