import tensorflow as tf
from keras.layers import Reshape,InputLayer

from get_model import get_model
from config_reader import config_test_reader


if __name__ == '__main__':

    params_test,params_model = config_test_reader()
    model = get_model(params_test,params_model)
    model.load_weights(params_model['keras_model_weights'])
    input_layer = model.get_layer(name='image')
    # input_layer.
    paf_layer = model.get_layer(name='Mconv7_stage6_L1')
    heat_layer = model.get_layer(name='Mconv7_stage6_L2')
    pafs = paf_layer.output
    heap_maps = heat_layer.output
    # pafs = tf.Session().run(pafs)
    print(pafs)
    print (model.summary())