from configobj import ConfigObj
import numpy as np


def config_test_reader():
    config = ConfigObj('config_test')

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['padValue'] = int(model['padValue'])
    model['batch_size'] = int(model['batch_size'])

    param['stride'] = int(param['stride'])
    param['crop_size_x'] = int(param['crop_size_x'])
    param['crop_size_y'] = int(param['crop_size_y'])
    param['np'] = int(param['np'])
    param['octave'] = int(param['octave'])
    # param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = map(float, param['scale_search'])
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    # param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model

def config_train_reader():
    config = ConfigObj('config_train')

    params_transform = config['params_transform']
    params_transform = dict(params_transform)
    params_transform = { key:float(value) for key,value in params_transform.items()}
    params_transform['mode'] = int(params_transform['mode'])
    params_transform['crop_size_x'] = int(params_transform['crop_size_x'])
    params_transform['crop_size_y'] = int(params_transform['crop_size_y'])
    params_transform['np'] = int(params_transform['np'])
    params_transform['stride'] = int(params_transform['stride'])

    # transform = [ for key,value in transform.]
    params_train = config['params_train']
    params_train['batch_size'] = int(params_train['batch_size'])
    params_train['steps_per_epoch'] = int(params_train['steps_per_epoch'])
    params_train['epochs'] = int(params_train['epochs'])

    return params_transform,params_train

if __name__ == "__main__":
    transform = config_train_reader()
    # transform = dict(transform)
    # print(type(transform))
    # print(transform)
    # transform = dict(transform)
    print transform
