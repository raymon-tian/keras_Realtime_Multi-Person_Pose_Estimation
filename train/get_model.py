from keras.models import Model
from keras.layers import Conv2D,Input,MaxPool2D,ZeroPadding2D,concatenate,Lambda,multiply,add,Reshape
from keras.initializers import RandomNormal,Constant
from keras.optimizers import SGD
import tensorflow as tf

from config_reader import config_train_reader

def get_model(params_transform,params_train):

    # params set
    batch_size = int(params_train['batch_size'])
    crop_size_x = int(params_transform['crop_size_x'])
    crop_size_y = int(params_transform['crop_size_y'])
    stride = int(params_transform['stride'])
    num_parts = int(params_transform['np'])
    grid_x = crop_size_x / stride
    grid_y = crop_size_y / stride

    stage_num = 6

    # input output
    image = Input(shape=(crop_size_y,crop_size_x,3),
                  batch_shape=(batch_size,crop_size_y,crop_size_x,3),
                  name='image')
    label = Input(shape=(grid_y,grid_x,(num_parts + 1)*2),
                  batch_shape=(batch_size,grid_y,grid_x,(num_parts+1)*2),
                  name='label')

    net_input = [image,label]
    net_output = []

    # ground-truth

    paf_weight = Lambda(lambda x: x[:, :, :, :38])(label)# mask
    confid_weight = Lambda(lambda x: x[:,:, :, 38:57])(label)# mask
    paf_temp = Lambda(lambda x: x[:,:, :, 57:95])(label)# gt
    confid_temp = Lambda(lambda x: x[:,:, :, 95:114])(label)#gt

    gt = concatenate([paf_temp, confid_temp], axis=3,name='ground-truth')

    # temp = concatenate([paf_weight,confid_weight],axis=3)
    # print(temp.shape)
    # common op
    image_padded = ZeroPadding2D()(image)
    conv1_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(),name='conv1_1')(image_padded)
    conv1_1_padded = ZeroPadding2D()(conv1_1)
    conv1_2 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(),name='conv1_2')(conv1_1_padded)
    pool1_stage1 = MaxPool2D(pool_size=(2,2),strides=2,name='pool1_stage1')(conv1_2)

    pool1_stage1_padded = ZeroPadding2D()(pool1_stage1)
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv2_1')(pool1_stage1_padded)
    conv2_1_padded = ZeroPadding2D()(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv2_2')(conv2_1_padded)
    # conv2_2 = ZeroPadding2D()(conv2_2)
    pool2_stage1 = MaxPool2D(pool_size=(2,2),strides=2,name='pool2_stage1')(conv2_2)

    pool2_stage1_padded = ZeroPadding2D()(pool2_stage1)
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_1')(pool2_stage1_padded)
    conv3_1_padded = ZeroPadding2D()(conv3_1)
    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_2')(conv3_1_padded)
    conv3_2_padded = ZeroPadding2D()(conv3_2)
    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_3')(conv3_2_padded)
    conv3_3_padded = ZeroPadding2D()(conv3_3)
    conv3_4 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_4')(conv3_3_padded)

    pool3_stage1 = MaxPool2D(pool_size=(2, 2), strides=2,  name='pool3_stage1')(conv3_4)
    pool3_stage1_padded = ZeroPadding2D()(pool3_stage1)
    conv4_1 = Conv2D(filters=512, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_1')(pool3_stage1_padded)
    conv4_1_padded = ZeroPadding2D()(conv4_1)
    conv4_2 = Conv2D(filters=512, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_2')(conv4_1_padded)
    conv4_2_padded = ZeroPadding2D()(conv4_2)
    conv4_3_CPM = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_3_CPM')(conv4_2_padded)
    conv4_3_CPM_padded = ZeroPadding2D()(conv4_3_CPM)

    conv4_4_CPM = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_4_CPM')(conv4_3_CPM_padded)

    # stage 1
    # L2 confidence maps
    conv4_4_CPM_padded = ZeroPadding2D()(conv4_4_CPM)
    conv5_1_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv5_1_CPM_L2')(conv4_4_CPM_padded)
    conv5_1_CPM_L2_padded = ZeroPadding2D()(conv5_1_CPM_L2)
    conv5_2_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_2_CPM_L2')(conv5_1_CPM_L2_padded)
    conv5_2_CPM_L2_padded = ZeroPadding2D()(conv5_2_CPM_L2)
    conv5_3_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_3_CPM_L2')(conv5_2_CPM_L2_padded)
    conv5_3_CPM_L2_padded = ZeroPadding2D(padding=(0,0))(conv5_3_CPM_L2)

    conv5_4_CPM_L2 = Conv2D(filters=512, kernel_size=(1, 1),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_4_CPM_L2')(conv5_3_CPM_L2_padded)
    conv5_4_CPM_L2_padded = ZeroPadding2D(padding=(0,0))(conv5_4_CPM_L2)

    conv5_5_CPM_L2 = Conv2D(filters=19, kernel_size=(1, 1),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_5_CPM_L2')(conv5_4_CPM_L2_padded)
    # L1 PAFs
    # conv4_4_CPM_padded = ZeroPadding2D()(conv4_4_CPM)
    conv5_1_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_1_CPM_L1')(conv4_4_CPM_padded)
    conv5_1_CPM_L1_padded = ZeroPadding2D()(conv5_1_CPM_L1)
    conv5_2_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_2_CPM_L1')(conv5_1_CPM_L1_padded)
    conv5_2_CPM_L1_padded = ZeroPadding2D()(conv5_2_CPM_L1)
    conv5_3_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_3_CPM_L1')(conv5_2_CPM_L1_padded)
    conv5_3_CPM_L1_padded = ZeroPadding2D(padding=(0, 0))(conv5_3_CPM_L1)

    conv5_4_CPM_L1 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_4_CPM_L1')(conv5_3_CPM_L1_padded)
    conv5_4_CPM_L1_padded = ZeroPadding2D(padding=(0, 0))(conv5_4_CPM_L1)

    conv5_5_CPM_L1 = Conv2D(filters=38, kernel_size=(1, 1), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_5_CPM_L1')(conv5_4_CPM_L1_padded)

    paf_masked_stage1_L1 = multiply([conv5_5_CPM_L1,paf_weight],name='paf_masked_stage1_L1')
    confid_masked_stage1_L2 = multiply([conv5_5_CPM_L2,confid_weight],name='confid_masked_stage1_L2')

    pred_label_stage1 = concatenate([paf_masked_stage1_L1,confid_masked_stage1_L2],axis=3,name='s1')
    pred_label_stage1 = Lambda(lambda x:tf.multiply(x,-1.0))(pred_label_stage1)
    pred_label_stage1 = add([pred_label_stage1,gt])
    pred_label_stage1 = Lambda(lambda x:tf.square(x))(pred_label_stage1)
    loss1 = Lambda(lambda x:tf.reduce_sum(x,axis=[1,2,3],keep_dims=True),name="scalar_s1")(pred_label_stage1)
    loss1 = Reshape((1,),name='final_s1')(loss1)
    # net_output.append(paf_masked_stage1_L1)
    # net_output.append(confid_masked_stage1_L2)
    net_output.append(loss1)

    temp_L1 = conv5_5_CPM_L1
    temp_L2 = conv5_5_CPM_L2

    # model = Model(inputs=image,outputs=[temp_L2,temp_L1])
    # return model

    for i in range(2,stage_num+1):
        concat_stagei = concatenate([temp_L1,temp_L2,conv4_4_CPM],axis=3)
        # L1
        concat_stage_padded = ZeroPadding2D(padding=(3,3))(concat_stagei)
        Mconv1_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='Mconv1_stage%s_L1'%(str(i)))(concat_stage_padded)
        Mconv1_stagei_L1_padded = ZeroPadding2D(padding=(3,3))(Mconv1_stagei_L1)
        Mconv2_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv2_stage%s_L1'%(str(i)))(Mconv1_stagei_L1_padded)
        Mconv2_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv2_stagei_L1)
        Mconv3_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv3_stage%s_L1'%(str(i)))(Mconv2_stagei_L1_padded)
        Mconv3_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv3_stagei_L1)
        Mconv4_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv4_stage%s_L1'%(str(i)))(Mconv3_stagei_L1_padded)
        Mconv4_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv4_stagei_L1)
        Mconv5_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv5_stage%s_L1'%(str(i)))(Mconv4_stagei_L1_padded)
        Mconv5_stagei_L1_padded = ZeroPadding2D(padding=(0, 0))(Mconv5_stagei_L1)
        Mconv6_stagei_L1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv6_stage%s_L1'%(str(i)))(Mconv5_stagei_L1_padded)
        Mconv6_stagei_L1_padded = ZeroPadding2D(padding=(0, 0))(Mconv6_stagei_L1)
        Mconv7_stagei_L1 = Conv2D(filters=38, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv7_stage%s_L1'%(str(i)))(Mconv6_stagei_L1_padded)

        # L2
        Mconv1_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv1_stage%s_L2'%(str(i)))(concat_stage_padded)
        Mconv1_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv1_stagei_L2)
        Mconv2_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv2_stage%s_L2'%(str(i)))(Mconv1_stagei_L2_padded)
        Mconv2_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv2_stagei_L2)
        Mconv3_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv3_stage%s_L2'%(str(i)))(Mconv2_stagei_L2_padded)
        Mconv3_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv3_stagei_L2)
        Mconv4_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv4_stage%s_L2'%(str(i)))(Mconv3_stagei_L2_padded)
        Mconv4_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv4_stagei_L2)
        Mconv5_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv5_stage%s_L2'%(str(i)))(Mconv4_stagei_L2_padded)
        Mconv5_stagei_L2_padded = ZeroPadding2D(padding=(0, 0))(Mconv5_stagei_L2)
        Mconv6_stagei_L2 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv6_stage%s_L2'%(str(i)))(Mconv5_stagei_L2_padded)
        Mconv6_stagei_L2_padded = ZeroPadding2D(padding=(0, 0))(Mconv6_stagei_L2)
        Mconv7_stagei_L2 = Conv2D(filters=19, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv7_stage%s_L2'%(str(i)))(Mconv6_stagei_L2_padded)

        temp_L1 = Mconv7_stagei_L1
        temp_L2 = Mconv7_stagei_L2

        paf_masked_stagei_L1 = multiply([temp_L1, paf_weight],name='paf_masked_stage%s_L1'%(str(i)))
        confid_masked_stagei_L2 = multiply([temp_L2, confid_weight],name='confid_masked_stage%s_L2'%(str(i)))

        pred_label_stagei = concatenate([paf_masked_stagei_L1, confid_masked_stagei_L2], axis=3,
                                        name='s%s'%(str(i)))

        pred_label_stagei = Lambda(lambda x:tf.multiply(x,-1.0))(pred_label_stagei)
        pred_label_stagei = add([pred_label_stagei,gt])
        pred_label_stagei = Lambda(lambda x:tf.square(x))(pred_label_stagei)
        # pred_label_stagei = Lambda(lambda x:tf.reduce_sum(x))(pred_label_stagei)
        lossi = Lambda(lambda x:tf.reduce_sum(x,axis=[1,2,3],keep_dims=True),name="scalar_s%s"%(str(i)))(pred_label_stagei)
        lossi = Reshape((1,), name='final_s%s'%(str(i)))(lossi)

        # net_output.append(paf_masked_stagei_L1)
        # net_output.append(confid_masked_stagei_L2)
        net_output.append(lossi)

    model = Model(inputs=net_input,outputs=net_output)
    model.compile(optimizer=SGD(lr=0.000040,momentum=0.9,decay=0.0005),
                  loss='mean_absolute_error')
    return model


def get_model_test(params):

    # params set
    batch_size = int(params['batch_size'])
    crop_size_x = int(params['crop_size_x'])
    crop_size_y = int(params['crop_size_y'])
    stride = int(params['stride'])
    np = int(params['np'])
    grid_x = crop_size_x / stride
    grid_y = crop_size_y / stride

    stage_num = 6

    # input
    image = Input(shape=(crop_size_y,crop_size_x,3),
                  batch_shape=(batch_size,crop_size_y,crop_size_x,3),
                  name='image')
    # common op
    image_padded = ZeroPadding2D()(image)
    conv1_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(),name='conv1_1')(image_padded)
    conv1_1_padded = ZeroPadding2D()(conv1_1)
    conv1_2 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(),name='conv1_2')(conv1_1_padded)
    pool1_stage1 = MaxPool2D(pool_size=(2,2),strides=2,name='pool1_stage1')(conv1_2)

    pool1_stage1_padded = ZeroPadding2D()(pool1_stage1)
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv2_1')(pool1_stage1_padded)
    conv2_1_padded = ZeroPadding2D()(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv2_2')(conv2_1_padded)
    # conv2_2 = ZeroPadding2D()(conv2_2)
    pool2_stage1 = MaxPool2D(pool_size=(2,2),strides=2,name='pool2_stage1')(conv2_2)

    pool2_stage1_padded = ZeroPadding2D()(pool2_stage1)
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_1')(pool2_stage1_padded)
    conv3_1_padded = ZeroPadding2D()(conv3_1)
    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_2')(conv3_1_padded)
    conv3_2_padded = ZeroPadding2D()(conv3_2)
    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_3')(conv3_2_padded)
    conv3_3_padded = ZeroPadding2D()(conv3_3)
    conv3_4 = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv3_4')(conv3_3_padded)

    pool3_stage1 = MaxPool2D(pool_size=(2, 2), strides=2,  name='pool3_stage1')(conv3_4)
    pool3_stage1_padded = ZeroPadding2D()(pool3_stage1)
    conv4_1 = Conv2D(filters=512, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_1')(pool3_stage1_padded)
    conv4_1_padded = ZeroPadding2D()(conv4_1)
    conv4_2 = Conv2D(filters=512, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_2')(conv4_1_padded)
    conv4_2_padded = ZeroPadding2D()(conv4_2)
    conv4_3_CPM = Conv2D(filters=256, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_3_CPM')(conv4_2_padded)
    conv4_3_CPM_padded = ZeroPadding2D()(conv4_3_CPM)

    conv4_4_CPM = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv4_4_CPM')(conv4_3_CPM_padded)

    # stage 1
    # L2 confidence maps
    conv4_4_CPM_padded = ZeroPadding2D()(conv4_4_CPM)
    conv5_1_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer=RandomNormal(stddev=0.00999999977648),
                     bias_initializer=Constant(), name='conv5_1_CPM_L2')(conv4_4_CPM_padded)
    conv5_1_CPM_L2_padded = ZeroPadding2D()(conv5_1_CPM_L2)
    conv5_2_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_2_CPM_L2')(conv5_1_CPM_L2_padded)
    conv5_2_CPM_L2_padded = ZeroPadding2D()(conv5_2_CPM_L2)
    conv5_3_CPM_L2 = Conv2D(filters=128, kernel_size=(3, 3),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_3_CPM_L2')(conv5_2_CPM_L2_padded)
    conv5_3_CPM_L2_padded = ZeroPadding2D(padding=(0,0))(conv5_3_CPM_L2)

    conv5_4_CPM_L2 = Conv2D(filters=512, kernel_size=(1, 1),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_4_CPM_L2')(conv5_3_CPM_L2_padded)
    conv5_4_CPM_L2_padded = ZeroPadding2D(padding=(0,0))(conv5_4_CPM_L2)

    conv5_5_CPM_L2 = Conv2D(filters=19, kernel_size=(1, 1),  activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_5_CPM_L2')(conv5_4_CPM_L2_padded)
    # L1 PAFs
    # conv4_4_CPM_padded = ZeroPadding2D()(conv4_4_CPM)
    conv5_1_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_1_CPM_L1')(conv4_4_CPM_padded)
    conv5_1_CPM_L1_padded = ZeroPadding2D()(conv5_1_CPM_L1)
    conv5_2_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_2_CPM_L1')(conv5_1_CPM_L1_padded)
    conv5_2_CPM_L1_padded = ZeroPadding2D()(conv5_2_CPM_L1)
    conv5_3_CPM_L1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_3_CPM_L1')(conv5_2_CPM_L1_padded)
    conv5_3_CPM_L1_padded = ZeroPadding2D(padding=(0, 0))(conv5_3_CPM_L1)

    conv5_4_CPM_L1 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_4_CPM_L1')(conv5_3_CPM_L1_padded)
    conv5_4_CPM_L1_padded = ZeroPadding2D(padding=(0, 0))(conv5_4_CPM_L1)

    conv5_5_CPM_L1 = Conv2D(filters=38, kernel_size=(1, 1), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='conv5_5_CPM_L1')(conv5_4_CPM_L1_padded)
    temp_L1 = conv5_5_CPM_L1
    temp_L2 = conv5_5_CPM_L2

    # model = Model(inputs=image,outputs=[temp_L2,temp_L1])
    # return model

    for i in range(2,stage_num+1):
        concat_stagei = concatenate([temp_L1,temp_L2,conv4_4_CPM],axis=3)
        # L1
        concat_stage_padded = ZeroPadding2D(padding=(3,3))(concat_stagei)
        Mconv1_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                            kernel_initializer=RandomNormal(stddev=0.00999999977648),
                            bias_initializer=Constant(), name='Mconv1_stage%s_L1'%(str(i)))(concat_stage_padded)
        Mconv1_stagei_L1_padded = ZeroPadding2D(padding=(3,3))(Mconv1_stagei_L1)
        Mconv2_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv2_stage%s_L1'%(str(i)))(Mconv1_stagei_L1_padded)
        Mconv2_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv2_stagei_L1)
        Mconv3_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv3_stage%s_L1'%(str(i)))(Mconv2_stagei_L1_padded)
        Mconv3_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv3_stagei_L1)
        Mconv4_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv4_stage%s_L1'%(str(i)))(Mconv3_stagei_L1_padded)
        Mconv4_stagei_L1_padded = ZeroPadding2D(padding=(3, 3))(Mconv4_stagei_L1)
        Mconv5_stagei_L1 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv5_stage%s_L1'%(str(i)))(Mconv4_stagei_L1_padded)
        Mconv5_stagei_L1_padded = ZeroPadding2D(padding=(0, 0))(Mconv5_stagei_L1)
        Mconv6_stagei_L1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv6_stage%s_L1'%(str(i)))(Mconv5_stagei_L1_padded)
        Mconv6_stagei_L1_padded = ZeroPadding2D(padding=(0, 0))(Mconv6_stagei_L1)
        Mconv7_stagei_L1 = Conv2D(filters=38, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv7_stage%s_L1'%(str(i)))(Mconv6_stagei_L1_padded)

        # L2
        Mconv1_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv1_stage%s_L2'%(str(i)))(concat_stage_padded)
        Mconv1_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv1_stagei_L2)
        Mconv2_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv2_stage%s_L2'%(str(i)))(Mconv1_stagei_L2_padded)
        Mconv2_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv2_stagei_L2)
        Mconv3_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv3_stage%s_L2'%(str(i)))(Mconv2_stagei_L2_padded)
        Mconv3_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv3_stagei_L2)
        Mconv4_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv4_stage%s_L2'%(str(i)))(Mconv3_stagei_L2_padded)
        Mconv4_stagei_L2_padded = ZeroPadding2D(padding=(3, 3))(Mconv4_stagei_L2)
        Mconv5_stagei_L2 = Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv5_stage%s_L2'%(str(i)))(Mconv4_stagei_L2_padded)
        Mconv5_stagei_L2_padded = ZeroPadding2D(padding=(0, 0))(Mconv5_stagei_L2)
        Mconv6_stagei_L2 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv6_stage%s_L2'%(str(i)))(Mconv5_stagei_L2_padded)
        Mconv6_stagei_L2_padded = ZeroPadding2D(padding=(0, 0))(Mconv6_stagei_L2)
        Mconv7_stagei_L2 = Conv2D(filters=19, kernel_size=(1, 1), activation='relu',
                                  kernel_initializer=RandomNormal(stddev=0.00999999977648),
                                  bias_initializer=Constant(), name='Mconv7_stage%s_L2'%(str(i)))(Mconv6_stagei_L2_padded)

        temp_L1 = Mconv7_stagei_L1
        temp_L2 = Mconv7_stagei_L2




    model = Model(inputs=image,outputs=[temp_L2,temp_L1])

    return model
if __name__ == '__main__':
    params_transform,params_train = config_train_reader()
    # print(params)
    model = get_model(params_transform,params_train)
    print(model.summary())

