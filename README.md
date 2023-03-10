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

## QAT per tensor

### Training 

With 60 data samples from `infos = infos[3624:3684]`

```
python train_qat.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth --batch_size 4 \
--output_dir ../output/leishen_models/qat_per_tensor --epochs 2
```

output:
```
Car:70.2122
Truck:69.5722
Bus:0.0000
Non_motor_vehicles:8.5712
Pedestrians:0.0000
```

### Testing

```
python test_qat.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--ckpt_qat ../output/leishen_models/qat_per_tensor/ckpt/checkpoint_epoch_2.pth \
--batch_size 4 --output_dir ../output/leishen_models/qat_conv_no_quant
```

output:
```
Car:70.2122
Truck:69.5722
Bus:0.0000
Non_motor_vehicles:8.5712
Pedestrians:0.0000
```

## QAT per channel

### Training 

With 60 data samples from `infos = infos[3624:3684]`

```
python train_qat.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth --batch_size 4 \
--output_dir ../output/leishen_models/qat_per_channel --epochs 2
```

output:
```
Car:71.9900
Truck:69.9686
Bus:0.0000
Non_motor_vehicles:7.6891
Pedestrians:0.0000
```

### Testing

```
python test_qat.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml \
--ckpt ../checkpoints/leishen/checkpoint_epoch_30.pth \
--ckpt_qat ../output/leishen_models/qat_per_channel/ckpt/checkpoint_epoch_2.pth \
--batch_size 4 --output_dir ../output/leishen_models/qat_conv_no_quant
```

output:
```
Car:71.9900
Truck:69.9686
Bus:0.0000
Non_motor_vehicles:7.6891
Pedestrians:0.0000
```

## QAT on 95 server

### Training

```
bash ~/ls_qat_1.sh
```

In the "jasonchen/leishen_qat" branch

```
python train_qat_hongbing.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --batch_size 24 --output_dir ../output_hongbing --epochs 2 
```

start epoch is 0

```  
Epoch 0 Average Loss=8.8152
Epoch 1 Average Loss=8.0318
```

### Testing

```
python test_qat_hongbing.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --ckpt_qat ../output_hongbing/ckpt/checkpoint_epoch_2.pth --batch_size 24 --output_dir ../output_hongbing
```

```
Car:70.6147
Truck:65.0209
Bus:14.6141
Non_motor_vehicles:37.3392
Pedestrians:0.0000
```

### Conversion with BSTNNX

copy `quant_param_dict.jason` and `torch_frozen_model.onnx`

```
bstnnx_run --config /bsnn/users/hongbing/tests/PointPillar/job.yaml --result_dir /bsnn/users/hongbing/tests/PointPillar/result --extra priority_range=100-300 --extra resume_from_breakpoint=False
```

copy 300 stage output `quant_model.onnx`

### ONNX Testing

switch to branch: `jasonchen/leishen_onnx_a1000b0`

```
python setup.py develop
```

```
python test.py --cfg_file ../tools/cfgs/leishen_models/pp_robosense_baseline_onnx.yaml --batch_size 1 --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth
```

```
Car:70.6147
Truck:65.0209
Bus:14.6141
Non_motor_vehicles:37.3392
Pedestrians:0.0000
```

## QAT per-channel on 95 server

### Training

```
bash ~/ls_qat_1.sh
```

In the "jasonchen/leishen_qat" branch

```
python train_qat_hongbing.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --batch_size 24 --output_dir ../output_hongbing_per_channel --epochs 2 
```

start epoch is 0

```  
Epoch 0 Average Loss=8.8152
Epoch 1 Average Loss=8.0318
```

### Testing

```
python test_qat_hongbing.py --config cfgs/leishen_models/pp_robosense_baseline_qat_0.yaml --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth --ckpt_qat ../output_hongbing/ckpt/checkpoint_epoch_2.pth --batch_size 24 --output_dir ../output_hongbing
```

```
Car:70.6147
Truck:65.0209
Bus:14.6141
Non_motor_vehicles:37.3392
Pedestrians:0.0000
```

### Conversion with BSTNNX

copy `quant_param_dict.jason` and `torch_frozen_model.onnx`

```
bstnnx_run --config /bsnn/users/hongbing/tests/PointPillar/job.yaml --result_dir /bsnn/users/hongbing/tests/PointPillar/result --extra priority_range=100-300 --extra resume_from_breakpoint=False
```

copy 300 stage output `quant_model.onnx`

### ONNX Testing

switch to branch: `jasonchen/leishen_onnx_a1000b0`

```
python setup.py develop
```

```
python test.py --cfg_file ../tools/cfgs/leishen_models/pp_robosense_baseline_onnx.yaml --batch_size 1 --ckpt ../output/tools/cfgs/leishen_models/pp_robosense_baseline_test/default/ckpt/checkpoint_epoch_30.pth
```

```
Car:70.6147
Truck:65.0209
Bus:14.6141
Non_motor_vehicles:37.3392
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

## Quantized Model without post activation quantization after Conv if ReLU is followed

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
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.13049905002117157, max_val=0.09413600713014603)
          )
          (activation_post_process): Identity()
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=31.566993713378906)
          )
        )
        (3): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.30858683586120605, max_val=0.2377110868692398)
          )
          (activation_post_process): Identity()
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=22.56749725341797)
          )
        )
        (6): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.33979588747024536, max_val=0.3424706757068634)
          )
          (activation_post_process): Identity()
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=30.97944450378418)
          )
        )
        (9): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.001953125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.23827292025089264, max_val=0.2375713437795639)
          )
          (activation_post_process): Identity()
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=20.431049346923828)
          )
        )
      )
      (1): Sequential(
        (0): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.23520724475383759, max_val=0.2521500289440155)
          )
          (activation_post_process): Identity()
        )
        (1): Identity()
        (2): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=21.96257209777832)
          )
        )
        (3): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.001953125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.21215786039829254, max_val=0.1797364205121994)
          )
          (activation_post_process): Identity()
        )
        (4): Identity()
        (5): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=13.37221622467041)
          )
        )
        (6): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.32819193601608276, max_val=0.24782663583755493)
          )
          (activation_post_process): Identity()
        )
        (7): Identity()
        (8): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=9.155134201049805)
          )
        )
        (9): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.3263762295246124, max_val=0.26731523871421814)
          )
          (activation_post_process): Identity()
        )
        (10): Identity()
        (11): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.464939594268799)
          )
        )
        (12): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.36335405707359314, max_val=0.3596310317516327)
          )
          (activation_post_process): Identity()
        )
        (13): Identity()
        (14): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=5.997600555419922)
          )
        )
        (15): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.00390625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.3433186411857605, max_val=0.2969321608543396)
          )
          (activation_post_process): Identity()
        )
        (16): Identity()
        (17): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=5.869157314300537)
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
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=19.918087005615234)
          )
        )
        (1): Conv2d(
          64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0009765625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.07263323664665222, max_val=0.0555475614964962)
          )
          (activation_post_process): Identity()
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.015625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.197892189025879)
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
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=5.8130388259887695)
          )
        )
        (1): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
          (weight_fake_quant): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0009765625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.08706443756818771, max_val=0.08933641016483307)
          )
          (activation_post_process): Identity()
        )
        (2): Identity()
        (3): ReLU(
          (activation_post_process): FakeQuantize(
            fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.015625, zero_point=0
            (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.197892189025879)
          )
        )
      )
    )
    (encoder): Sequential(
      (0): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.125, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-13.345178604125977, max_val=11.107205390930176)
        )
        (activation_post_process): Identity()
      )
      (1): Identity()
      (2): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=39.49658966064453)
        )
      )
      (3): Conv2d(
        128, 128, kernel_size=(1, 1), stride=(1, 1)
        (weight_fake_quant): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.846174955368042, max_val=3.8979547023773193)
        )
        (activation_post_process): Identity()
      )
      (4): Identity()
      (5): ReLU(
        (activation_post_process): FakeQuantize(
          fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
          (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=61.55551528930664)
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
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.015625, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.197892189025879)
      )
    )
    (quant): QuantStub(
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-3.499983310699463, max_val=2.860299587249756)
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
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.648526668548584, max_val=1.8285213708877563)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.5, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-43.816505432128906, max_val=3.4075369834899902)
      )
    )
    (conv_box): Conv2d(
      256, 70, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0078125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.8128230571746826, max_val=0.7450146675109863)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.03125, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-2.454333543777466, max_val=1.4543596506118774)
      )
    )
    (conv_dir_cls): Conv2d(
      256, 40, kernel_size=(1, 1), stride=(1, 1)
      (weight_fake_quant): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.0625, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-5.742462158203125, max_val=3.394367218017578)
      )
      (activation_post_process): FakeQuantize(
        fake_quant_enabled=tensor([1], device='cuda:0', dtype=torch.uint8), observer_enabled=tensor([0], device='cuda:0', dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=0.25, zero_point=0
        (activation_post_process): MovingAverageMinMaxObserver(min_val=-31.13861656188965, max_val=19.237991333007812)
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
