import numpy as np
import torch
import sys
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops


def convert_keys(model):
    converted = {}
    for k, v in model.items():
        new_k = k[len('decoder.'):].replace('layer.', 'layer_')
        if new_k.endswith('.weight'):
            replacement = '' if new_k.endswith('embeddings.weight') else '.kernel'
            new_k = new_k.replace('.weight', replacement)
        new_k = new_k.replace('.', '/')
        new_k = new_k.replace('cls/predictions/bias', 'cls/predictions/output_bias')
        new_k = new_k.replace('cls/seq_relationship/bias', 'cls/seq_relationship/output_bias')
        new_k = new_k.replace('cls/seq_relationship/kernel', 'cls/seq_relationship/output_weights')
        converted[new_k] = v
    return converted


def load_from_pytorch_checkpoint(checkpoint, assignment_map):
    pytorch_model = torch.load(checkpoint, map_location='cpu')['model']
    pt_model_with_tf_keys = convert_keys(pytorch_model)
    for _, name in assignment_map.items():
        store_vars = vs._get_default_variable_store()._vars
        var = store_vars.get(name, None)
        assert var is not None
        if name not in pt_model_with_tf_keys:
            print('WARNING:', name, 'not found in original model.')
            continue
        array = pt_model_with_tf_keys[name].cpu().numpy()
        if name.endswith('kernel'):
            array = array.transpose()
        assert tuple(var.get_shape().as_list()) == tuple(array.shape)
        init_value = ops.convert_to_tensor(array, dtype=np.float32)
        var._initial_value = init_value
        var._initializer_op = var.assign(init_value)


def print_vars(pytorch_ckpt, tf_ckpt):
    tf_vars = tf.train.list_variables(tf_ckpt)
    tf_vars = {k:v for (k, v) in tf_vars}
    pytorch_model = torch.load(pytorch_ckpt)['model']
    pt_model_with_tf_keys = convert_keys(pytorch_model)
    only_pytorch, only_tf, common = [], [], []
    tf_only = set(tf_vars.keys())
    for k, v in pt_model_with_tf_keys.items():
        if k in tf_vars:
            common.append(k)
            tf_only.remove(k)
        else:
            only_pytorch.append(k)
    print('-------------------')
    print('Common', len(common))
    for k in common:
        array = pt_model_with_tf_keys[k].cpu().numpy()
        if k.endswith('kernel'):
            array = array.transpose()
        tf_shape = tuple(tf_vars[k])
        pt_shape = tuple(array.shape)
        if tf_shape != pt_shape:
            print(k, tf_shape, pt_shape)
    print('-------------------')
    print('Pytorch only', len(only_pytorch))
    for k in only_pytorch:
        print(k, pt_model_with_tf_keys[k].size())
    print('-------------------')
    print('TF only', len(tf_only))
    for k in tf_only:
        print(k, tf_vars[k])

if __name__ == '__main__':
    print_vars(sys.argv[1], sys.argv[2])
