import numpy as np
import torch
import sys
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops

tensors_to_transpose = (
        "dense/kernel",
        "attention/self/query",
        "attention/self/key",
        "attention/self/value"
    )

var_map = (
    ('layer.', 'layer_'),
    ('word_embeddings.weight', 'word_embeddings'),
    ('position_embeddings.weight', 'position_embeddings'),
    ('token_type_embeddings.weight', 'token_type_embeddings'),
    ('.', '/'),
    ('LayerNorm/weight', 'LayerNorm/gamma'),
    ('LayerNorm/bias', 'LayerNorm/beta'),
    ('weight', 'kernel')
)

def to_tf_var_name(name: str):
    for patt, repl in iter(var_map):
        name = name.replace(patt, repl)
    return '{}'.format(name)

def my_convert_keys(model):
    converted = {}
    for k_pt, v in model.items():
        k_tf =  to_tf_var_name(k_pt)
        converted[k_tf] = v
    return converted

def load_from_pytorch_checkpoint(checkpoint, assignment_map):
    pytorch_model = torch.load(checkpoint, map_location='cpu')
    pt_model_with_tf_keys = my_convert_keys(pytorch_model)
    for _, name in assignment_map.items():
        store_vars = vs._get_default_variable_store()._vars
        var = store_vars.get(name, None)
        assert var is not None
        if name not in pt_model_with_tf_keys:
            print('WARNING:', name, 'not found in original model.')
            continue
        array = pt_model_with_tf_keys[name].cpu().numpy()
        if any([x in name for x in tensors_to_transpose]):
            array = array.transpose()
        assert tuple(var.get_shape().as_list()) == tuple(array.shape)
        init_value = ops.convert_to_tensor(array, dtype=np.float32)
        var._initial_value = init_value
        var._initializer_op = var.assign(init_value)


def print_vars(pytorch_ckpt, tf_ckpt):
    tf_vars = tf.train.list_variables(tf_ckpt)
    tf_vars = {k:v for (k, v) in tf_vars}
    pytorch_model = torch.load(pytorch_ckpt)
    pt_model_with_tf_keys = my_convert_keys(pytorch_model)
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
        if any([x in k for x in tensors_to_transpose]):
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
