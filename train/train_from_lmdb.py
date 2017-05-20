#coding=utf-8
import lmdb
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import struct
import pprint
import cv2 as cv
import ConfigParser
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,ProgbarLogger
import json

from get_model import get_model
from COCOLmdb import COCOLmdb
from utils import visualize_body_part,mkdir
from config_reader import config_train_reader



def generate_arrays_from_file(params_transform,params_train):

    path = params_train['lmdb_path']
    batch_size = params_train['batch_size']

    lmdb_env = lmdb.open(path,readonly=True)
    with lmdb_env.begin() as lmdb_txn:
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()
        cnt = 0
        X = []
        Y = []
        print('load a new batch  ================================  \n')                                                                                                                                                                                                                                                                                                                                                                                                                                 

        for idx,(key, value) in enumerate(lmdb_cursor):
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            """
            print data.shape
            # print(type(data))
            # print label
            meta_data = data[3,:,:]
            # print(meta_data.dtype)
            # print(struct.unpack('10s',meta_data[0:10,:]))
            # print(meta_data.shape)
            # str = [ chr(meta_data[0][i]) for i in range(meta_data.shape[1])]
            # print str
            # str = ''.join(str).replace('\x00','')
            # print str,len(str)
            print struct.unpack('2f',meta_data[1,0:8])
            print struct.unpack('3B',meta_data[2,0:3])
            print struct.unpack('2f',meta_data[3, 0:8])
            print struct.unpack('1f',meta_data[4,0:4])[0]
            joints = []
            for i in range(3):
                temp = struct.unpack('17f', meta_data[5 + i, 0:17 * 4])
                print '*************'
                print temp
                joints.append(temp)

            r = zip(*joints)
            r = np.array(r)
            print r.shape
            print r
            """
            cocoImg = COCOLmdb(data,params_transform)
            # cocoImg.set_meta_data()
            cocoImg.add_neck()
            # cocoImg.visualize()
            cocoImg.aug_scale()
            # cocoImg.visualize()
            cocoImg.aug_croppad()
            # cocoImg.visualize()
            cocoImg.aug_flip()
            # cocoImg.visualize()
            cocoImg.set_ground_truth()
            sample,label = cocoImg.get_sample_label()
            # print(sample.shape,label.shape)
            # cocoImg.visualize_heat_maps()
            # cocoImg.visualize()
            # cocoImg.aug_scale()
            # cocoImg.aug_croppad()
            # img,anno,mask_miss,mask_all = cocoImg.get_meta_data()
            # print(type(img))
            # img = np.rollaxis(img,0,3)

            # cocoImg.add_neck()
            # pprint.pprint(cocoImg.anno)
            # visualize_body_part(img,anno['joint_others'])
            # print(anno['joint_others'].shape)
            # cocoImg.aug_scale()
            # cocoImg.visualize()
            X.append(sample)
            Y.append(label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                X = np.array(X)
                Y = np.array(Y)
                GT = [Y for i in range(6)]
                # print(X.shape,Y.shape)
                yield (dict(image=X,label=Y),GT)
                X = []
                Y = []

if __name__ == '__main__':

    params_transform,params_train = config_train_reader()
    genenor = generate_arrays_from_file(params_transform,params_train)
    r = genenor.next()
    print(len(r))
    name_experiment = params_train['name_experiment']
    # exit()
    batch_size = params_train['batch_size']
    mkdir(name_experiment)

    filepath = './' + name_experiment + '/' + name_experiment + '_weights_{epoch:02d}_{loss:.4f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1, monitor='loss', mode='min',
                                   save_best_only=True,
                                   save_weights_only=True)
    progbarLogger = ProgbarLogger()
    model = get_model(params_transform,params_train)

    """
    COCO数据集中
    train2014  : 82783 个样本
    val2014    : 40504 个样本
    数据集生成的时候，将val2014中的前2644个样本标记位 'isValidation = 1'
    所以用于训练的样本数为 82783+40504-2644 = 120643
    这里设置，我们训练model的总样本数为 6000000
    """

    model.fit_generator(generator=generate_arrays_from_file(params_transform,params_train),
                        steps_per_epoch=params_train['steps_per_epoch'],
                        epochs=params_train['epochs'],
                        verbose=1,
                        # callbacks=[checkpointer,progbarLogger]
                        callbacks=[checkpointer])

    # save the weights of the model
    model.save_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.hdf5', overwrite=True)
    # save the architecture of the net to a json file
    with open('./' + name_experiment + '/' + name_experiment + '_net.json','w') as net_json:
        net_json.write(json.dump(model.to_json()))


