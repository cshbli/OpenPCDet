import torch
import torch.nn as nn

# BIAS_DTYPE = 'int24'
BIAS_QUANT_MIN = -2 ** 23
BIAS_QUANT_MAX = 2 ** 23 - 1

def quantize_model(model, device, backend='default', sample_data=None):
    model.to(device)
    model.train()
    if backend == 'default':
        import torch.quantization as quantizer

        activation_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.default_observer.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        weight_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.default_observer.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)

        # assign qconfig to model
        model.qconfig = quantizer.QConfig(activation=activation_quant, weight=weight_quant)

        # prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)
    elif backend == 'bst':
        import bstnnx_training.PyTorch.QAT.core as quantizer

        # Conv is quantized with rounding on hardware
        bst_activation_quant_int8 = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        # Input is quantized with truncating on hardware
        bst_activation_quant_int8_truncate = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, rounding='truncate', qscheme=torch.per_tensor_affine, reduce_range=False)
        bst_activation_quant_uint8 = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MovingAverageMinMaxObserver.with_args(dtype=torch.quint8), 
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        bst_weight_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)

        # 1) [bst_alignment] get b0 pre-bind qconfig adjusting Conv's activation quant scheme        
        b0_pre_bind_qconfig = quantizer.pre_bind(model, input_tensor=sample_data.to('cpu'), debug_mode=True, observer_scheme_dict={"weight_scheme": "MovingAverageMinMaxObserver", "activation_scheme": "MovingAverageMinMaxObserver"})

        # 2) assign qconfig to model
        # model.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant, qconfig_dict=b0_pre_bind_qconfig)

        # Disable quantizations of all activations, such as after "Conv+BN", "Conv", "Concat" and "ReLU" etc.
        model.qconfig = quantizer.QConfig(activation=nn.Identity, weight=bst_weight_quant, qconfig_dict=b0_pre_bind_qconfig)

        # Enable INT8 and Truncate quantization for input
        model.backbone_2d.quant.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8_truncate, weight=bst_weight_quant)

        # Enable UINT8 quantizations after all ReLUs
        for relu_module_idx in [2, 5]:
            model.backbone_2d.encoder[relu_module_idx].qconfig = quantizer.QConfig(activation=bst_activation_quant_uint8, weight=bst_weight_quant)

        for relu_module_idx in [2, 5, 8, 11]:
            model.backbone_2d.blocks[0][relu_module_idx].qconfig = quantizer.QConfig(activation=bst_activation_quant_uint8, weight=bst_weight_quant)
        
        for relu_module_idx in [2, 5, 8, 11, 14, 17]:
            model.backbone_2d.blocks[1][relu_module_idx].qconfig = quantizer.QConfig(activation=bst_activation_quant_uint8, weight=bst_weight_quant)

        for relu_module_idx in [0, 1]:
            model.backbone_2d.deblocks[relu_module_idx][3].qconfig = quantizer.QConfig(activation=bst_activation_quant_uint8, weight=bst_weight_quant)

        # Enable INT8 quantizations after upsample conv and ConvTranspose2d
        model.backbone_2d.deblocks[0][0].qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        model.backbone_2d.deblocks[1][0].qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        model.backbone_2d.upsample.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        model.backbone_2d.upsample1.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        # model.backbone_2d.deblocks[0][0].qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=nn.Identity)
        # model.backbone_2d.deblocks[1][0].qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=nn.Identity)
        # model.backbone_2d.upsample.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=nn.Identity)
        # model.backbone_2d.upsample1.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=nn.Identity)

        # Enable UINT8 quantizations after Concat
        model.backbone_2d.cat1.qconfig = quantizer.QConfig(activation=bst_activation_quant_uint8, weight=bst_weight_quant)

        # Enable INT8 quantization after Convs of dense_head
        model.dense_head.conv_cls.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        model.dense_head.conv_box.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)
        model.dense_head.conv_dir_cls.qconfig = quantizer.QConfig(activation=bst_activation_quant_int8, weight=bst_weight_quant)

        # 3) prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)

        # 4) [bst_alignment] link model observers
        prepared_model = quantizer.link_modules(prepared_model, auto_detect=True, input_tensor=sample_data.to('cpu'), inplace=False, debug_mode=True)

    return prepared_model


def quantize_bias_np(bias, scale, zero_point, quant_min, quant_max):
    quant_t = bias / scale + zero_point
    clipped_t = np.clip(quant_t, quant_min, quant_max)
    round_t = np.round(clipped_t)
    return round_t.astype(np.float32)


def dequantize_bias_np(quantized_bias, scale, zero_point):
    float_tensor = quantized_bias.astype(np.float32)
    return ((float_tensor - zero_point) * scale).astype(np.float32)


def quantize_module_bias(module, input_module):
    input_scale = input_module.activation_post_process.scale
    weight_scale = module.weight_fake_quant.scale
    bias = module.bias
    bias_scale = input_scale * weight_scale
    bias_quant = bias / bias_scale
    bias_clip = torch.clip(bias_quant, BIAS_QUANT_MIN, BIAS_QUANT_MAX)
    bias_int = torch.round(bias_clip)
    bias_float = bias_int * bias_scale
    # module.bias.requires_grad = False
    # module.bias.copy_(bias_float)
    with torch.no_grad():
        module.bias.copy_(bias_float)
    module.bias_scale = bias_scale
    module.bias_zero_point = 0


def quantize_bias(model):
    quantize_module_bias(model.backbone_2d.encoder[0], model.backbone_2d.quant)
    quantize_module_bias(model.backbone_2d.encoder[3], model.backbone_2d.encoder[2])

    quantize_module_bias(model.backbone_2d.blocks[0][0], model.backbone_2d.encoder[5])
    quantize_module_bias(model.backbone_2d.blocks[0][3], model.backbone_2d.blocks[0][2])
    quantize_module_bias(model.backbone_2d.blocks[0][6], model.backbone_2d.blocks[0][5])
    quantize_module_bias(model.backbone_2d.blocks[0][9], model.backbone_2d.blocks[0][8])

    quantize_module_bias(model.backbone_2d.blocks[1][0], model.backbone_2d.blocks[0][11])
    quantize_module_bias(model.backbone_2d.blocks[1][3], model.backbone_2d.blocks[1][2])
    quantize_module_bias(model.backbone_2d.blocks[1][6], model.backbone_2d.blocks[1][5])
    quantize_module_bias(model.backbone_2d.blocks[1][9], model.backbone_2d.blocks[1][8])
    quantize_module_bias(model.backbone_2d.blocks[1][12], model.backbone_2d.blocks[1][11])
    quantize_module_bias(model.backbone_2d.blocks[1][15], model.backbone_2d.blocks[1][14])

    quantize_module_bias(model.backbone_2d.deblocks[0][1], model.backbone_2d.deblocks[0][0])
    quantize_module_bias(model.backbone_2d.deblocks[1][0], model.backbone_2d.blocks[1][17])
    quantize_module_bias(model.backbone_2d.upsample, model.backbone_2d.blocks[1][17])
    quantize_module_bias(model.backbone_2d.deblocks[1][1], model.backbone_2d.deblocks[1][0])

    quantize_module_bias(model.dense_head.conv_dir_cls, model.backbone_2d.cat1)
    quantize_module_bias(model.dense_head.conv_box, model.backbone_2d.cat1)
    quantize_module_bias(model.dense_head.conv_cls, model.backbone_2d.cat1)
