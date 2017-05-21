#coding=utf-8
import struct
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ma
from utils import visualize_body_part,visualize_img


class COCOLmdb(object):
    """
    
    %      ============= 实验中的索引形式   ==============
    %         (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
    %          5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'  
    %          9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee' 
    %          13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear' 
    %          17-'left_ear' )
    """

    def __init__(self,raw_data,transform):
        self.mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
        self.mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        self.raw_data = raw_data
        self.transform = transform
        self.aug_params = dict()
        self.__set_meta_data()

    def __set_meta_data(self):
        # 原始的BGR图像 (H,W,3)
        self.img = np.rollaxis(self.raw_data[0:3,:,:],0,3)
        # 图像对应的标注信息，标注信息存储在3-th slice中
        self.anno = self.__get_anno(self.raw_data[3,:,:])
        self.mask_miss = self.raw_data[4,:,:]
        self.mask_all = self.raw_data[5,:,:]

    def get_meta_data(self):
        return self.img,self.anno,self.mask_miss,self.mask_all

    def __get_string(self,line):
        ch_list = [chr(line[i]) for i in range(len(line))]
        str = ''.join(ch_list).replace('\x00','')
        return str

    def __get_anno(self,raw_anno):
        """
        该解析过程严格地对应genLMDB.py
        :param raw_anno: (H,W)表示一个二维矩阵，存储图像标注信息
        :return: 
        """
        anno = dict()
        # dataset name (string)
        anno['dataset'] = self.__get_string(raw_anno[0,:])
        # image height, image width
        anno['img_height'],anno['img_width'] = struct.unpack('2f',raw_anno[1,0:8])
        anno['img_height'] = int(anno['img_height'])
        anno['img_width'] = int(anno['img_width'])
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
        anno['isValidation'],anno['numOtherPeople'],anno['people_index'] = struct.unpack('3B',raw_anno[2,0:3])
        anno['annolist_index'],anno['writeCount'],anno['totalWriteCount'] = struct.unpack('3f',raw_anno[2,3:15])

        anno['annolist_index'] = int(anno['annolist_index'])
        anno['numOtherPeople'] = int(anno['numOtherPeople'])
        anno['people_index'] = int(anno['people_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(struct.unpack('2f',raw_anno[3,0:8]))
        # (c) scale_provided (float)
        anno['scale_provided'] = struct.unpack('1f',raw_anno[4,0:4])[0]
        # (d) joint_self (3*17) (float) (3 line)
        # 这里，原作者注释为 3*16 ，但是COCO数据集人体关键点是17，所以，斯认为是3*17
        joints = []
        for i in range(3):
            temp = struct.unpack('17f',raw_anno[5+i,0:17*4])
            joints.append(temp)
        joints = zip(*joints)
        anno['joint_self'] = np.array(joints)# 17×3（x,y,可视性标志,1表示进行了标记，并且可见；0表示进行了标记，但是不可见；2表示没有进行标记）
        # objpos_other_x (float), objpos_other_y (float) (nop lines)
        num_other_person = anno['numOtherPeople']
        row_idx = 8
        joint_other = []
        objpos_other = []
        scale_provided_other = []
        if(num_other_person!=0):
            # objpos_other_x (float), objpos_other_y (float) (num_other_person lines)
            for i in range(num_other_person):
                temp = struct.unpack('2f',raw_anno[row_idx+i,0:8])
                objpos_other.append(temp)
            row_idx += num_other_person
            # scale_provided_other (num_other_person floats in 1 line)
            scale_provided_other = np.array(struct.unpack('%sf'%(num_other_person),raw_anno[row_idx,0:4*num_other_person]))
            row_idx += 1
            # joint_others (3*17) (float) (num_other_person*3 lines) 原作者写着3*16，斯认为应该还是3*17
            for i in range(num_other_person):
                joint_other_one_p = []
                for j in range(3):
                    temp = struct.unpack('17f',raw_anno[row_idx+j,0:4*17])
                    joint_other_one_p.append(temp)
                row_idx += 3
                joint_other_one_p = np.array(zip(*joint_other_one_p))
                joint_other.append(joint_other_one_p)
            joint_other = np.array(joint_other)
            anno['joint_others'] = joint_other
            anno['objpos_other'] = np.array(objpos_other)
            anno['scale_provided_other'] = scale_provided_other
        return anno

    def add_neck(self):
        """
        COCO中有17个关键点，没有对neck的标注，需要将neck标注加进去
        :return: 
        """
        our_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # 索引6 5 分别表示 右肩 和 左肩
        right_shoulder = self.anno['joint_self'][6, :]
        left_shoulder = self.anno['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if (right_shoulder[2] != 2 or left_shoulder[2] != 2):
            neck[2] = 1
        else:
            neck[2] = 2
        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        self.anno['joint_self'] = np.vstack((self.anno['joint_self'], neck))
        self.anno['joint_self'] = self.anno['joint_self'][our_order,:]
        temp = []
        for i in range(self.anno['numOtherPeople']):
            right_shoulder = self.anno['joint_others'][i,6, :]
            left_shoulder = self.anno['joint_others'][i,5, :]
            neck = (right_shoulder + left_shoulder) / 2
            if (right_shoulder[2] != 2 or left_shoulder[2] != 2):
                neck[2] = 1
            else:
                neck[2] = 2
            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)
            single_p = np.vstack((self.anno['joint_others'][i], neck))
            single_p = single_p[our_order,:]
            temp.append(single_p)
        self.anno['joint_others'] = np.array(temp)

    def data_aug(self):

        pass
    def aug_scale(self):
        dice = random.random() # dice在[0,1]之间
        if(dice > self.transform['scale_prob']):
            # img_temp = self.img.copy()
            scale_multiplier = 1
        else:
            dice2 = random.random()
            # scale_multiplier 在 [0.9,1.1]之间
            scale_multiplier = (self.transform['scale_max'] - self.transform['scale_min']) * dice2 + self.transform['scale_min']
        scale_abs = self.transform['target_dist']/self.anno['scale_provided']
        scale = scale_abs * scale_multiplier
        self.img = cv.resize(self.img, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        if(self.transform['mode'] > 4):
            self.mask_miss = cv.resize(self.mask_miss,(0,0),fx=scale,fy=scale,interpolation=cv.INTER_CUBIC)
        if(self.transform['mode'] > 5):
            self.mask_all = cv.resize(self.mask_all, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        # modify meta data
        self.anno['objpos'] *= scale
        self.anno['joint_self'][:,:2] *= scale
        if(self.anno['numOtherPeople']!=0):
            self.anno['objpos_other'] *= scale
            self.anno['joint_others'][:,:,:2] *= scale

    def aug_rotate(self):
        mode = int(self.transform['mode'])

        dice = random.random()
        degree = (dice - 0.5)*2*self.transform['max_rotate_degree'] # degree [-10,10]
        center = (self.img.shape[1]/2.0,self.img.shape[0]/2.0)
        R = cv.getRotationMatrix2D(center=center,angle=degree,self=1.0)
        # bbox = cv.
        pass
    """
    def aug_croppad(self):
        # visualize_img(self.img)
        dice_x = random.random()
        dice_y = random.random()
        crop_x = int(self.transform['crop_size_x'])
        crop_y = int(self.transform['crop_size_y'])
        x_offset = int((dice_x - 0.5) * 2 * self.transform['center_perterb_max']) # [-10,10]
        y_offset = int((dice_y - 0.5) * 2 * self.transform['center_perterb_max']) # [-10,10]
        # print("*********")
        # print(self.anno['objpos'])
        center = self.anno['objpos'] + np.array([x_offset,y_offset])
        # print(center)
        offset_left = -(center[0] - (crop_x / 2));
        offset_up = -(center[1] - (crop_y / 2));
        self.img_crop = np.ones((crop_y,crop_x,3),dtype=np.uint8)*128
        self.mask_miss_crop = np.ones((crop_y,crop_x),dtype=np.uint8)*255
        self.mask_all_crop = np.zeros((crop_y,crop_x),dtype=np.uint8)
        if(crop_x%2 == 0):
            b_left = int(max((center[0]-crop_x/2,0)))
            b_right = int(min(( center[0] + crop_x/2,self.img.shape[1]-1)))
        else:
            b_left = int(max((center[0] - crop_x / 2, 0)))
            b_right = int(min((center[0] + crop_x / 2 + 1, self.img.shape[1])))
        if(crop_y%2 == 0):
            b_up = int(max((center[1]-crop_y/2, 0)))
            b_down = int(min((center[1]+crop_y/2, self.img.shape[0]-1)))
        else:
            b_up = int(max((center[1] - crop_y / 2, 0)))
            b_down = int(min((center[1] + crop_y / 2 + 1, self.img.shape[0])))
        # print(b_left,b_right,b_up,b_down)
        temp = self.img[b_up:b_down,b_left:b_right,:]
        # visualize_img(temp)
        # print(temp.shape)
        bias_h = 0
        bias_v = 0
        if(b_left == 0):
            bias_h = int(crop_x/2 - center[0])
        if(b_up == 0):
            bias_v = int(crop_y/2 -center[1])
        self.img_crop[0+bias_v:temp.shape[0]+bias_v,0+bias_h:temp.shape[1]+bias_h,:] = temp
        self.img = self.img_crop
        # visualize_img(self.img_crop)
        if(self.transform['mode'] > 4):
            temp = self.mask_miss[b_up:b_down,b_left:b_right]
            self.mask_miss_crop[0+bias_v:temp.shape[0]+bias_v,0+bias_h:temp.shape[1]+bias_h] = temp
            self.mask_miss = self.mask_miss_crop
        if(self.transform['mode'] > 5):
            temp = self.mask_all[b_up:b_down,b_left:b_right]
            self.mask_all_crop[0+bias_v:temp.shape[0]+bias_v,0+bias_h:temp.shape[1]+bias_h] = temp
            self.mask_all = self.mask_miss_crop

        # visualize_body_part(self.img, self.anno['joint_self'])
        # modify meta data
        offset_left += bias_h
        offset_up += bias_v
        self.aug_params['offset'] = (offset_up,offset_left)
        offset = np.array([offset_left,offset_up])
        self.anno['objpos'] += offset
        self.anno['joint_self'][:,:2] += offset
        if(self.anno['numOtherPeople']!=0):
            self.anno['objpos_other'] += offset
            self.anno['joint_others'][:,:,:2] += offset
        # visualize_body_part(self.img_crop, self.anno['joint_self'])
        # visualize_img(self.mask_miss_crop)
        # visualize_img(self.mask_all_crop)
    """
    def aug_croppad(self):

        dice_x = random.random()
        dice_y = random.random()
        crop_x = int(self.transform['crop_size_x'])
        crop_y = int(self.transform['crop_size_y'])
        x_offset = int((dice_x - 0.5) * 2 * self.transform['center_perterb_max']) # [-10,10]
        y_offset = int((dice_y - 0.5) * 2 * self.transform['center_perterb_max']) # [-10,10]

        center = self.anno['objpos'] + np.array([x_offset,y_offset])
        center = center.astype(int)
        # print(center)

        # pad up and down
        pad_v = np.ones((crop_y,self.img.shape[1],3),dtype=np.uint8)*128
        pad_v_mask_miss = np.ones((crop_y,self.mask_miss.shape[1]),dtype=np.uint8)*255
        pad_v_mask_all = np.zeros((crop_y,self.mask_all.shape[1]),dtype=np.uint8)
        self.img = np.concatenate((pad_v,self.img,pad_v),axis=0)
        self.mask_miss = np.concatenate((pad_v_mask_miss,self.mask_miss,pad_v_mask_miss),axis=0)
        self.mask_all = np.concatenate((pad_v_mask_all,self.mask_all,pad_v_mask_all),axis=0)
        # pad right and left
        pad_h = np.ones((self.img.shape[0],crop_x,3),dtype=np.uint8)*128
        pad_h_mask_miss = np.ones((self.mask_miss.shape[0],crop_x),dtype=np.uint8)*255
        pad_h_mask_all = np.zeros((self.mask_all.shape[0],crop_x),dtype=np.uint8)
        self.img = np.concatenate((pad_h,self.img,pad_h),axis=1)
        self.mask_miss = np.concatenate((pad_h_mask_miss,self.mask_miss,pad_h_mask_miss),axis=1)
        self.mask_all = np.concatenate((pad_h_mask_all,self.mask_all,pad_h_mask_all),axis=1)

        # visualize_img(self.img)
        # visualize_img(self.mask_miss)
        # visualize_img(self.mask_all)

        self.img = self.img[center[1]+crop_y/2:center[1]+crop_y/2+crop_y,center[0]+crop_x/2:center[0]+crop_x/2+crop_x,:]
        if(self.transform['mode'] > 4):
            self.mask_miss = self.mask_miss[center[1]+crop_y/2:center[1]+crop_y/2+crop_y+1,center[0]+crop_x/2:center[0]+crop_x/2+crop_x+1]
        if(self.transform['mode'] > 5):
            self.mask_all = self.mask_all[center[1]+crop_y/2:center[1]+crop_y/2+crop_y+1,center[0]+crop_x/2:center[0]+crop_x/2+crop_x+1]

        # visualize_img(self.img)
        # visualize_img(self.mask_miss)
        # visualize_img(self.mask_all)

        # offset_left = crop_x-center[0]+crop_x/2
        # offset_up = crop_y-center[1]+crop_y/2

        offset_left = crop_x/2 - center[0]
        offset_up = crop_y/2 - center[1]
        self.aug_params['offset'] = (offset_left,offset_up)
        offset = np.array([offset_left,offset_up])
        self.anno['objpos'] += offset
        self.anno['joint_self'][:,:2] += offset
        mask = np.logical_or.reduce((self.anno['joint_self'][:,0]>=crop_x,
                              self.anno['joint_self'][:,0]<0,
                              self.anno['joint_self'][:,1]>=crop_y,
                              self.anno['joint_self'][:,1]<0))
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        self.anno['joint_self'][mask==True,2] = 2
        if(self.anno['numOtherPeople']!=0):
            self.anno['objpos_other'] += offset
            self.anno['joint_others'][:,:,:2] += offset
            mask = np.logical_or.reduce((self.anno['joint_others'][:,:, 0] >= crop_x,
                                  self.anno['joint_others'][:,:, 0] < 0,
                                  self.anno['joint_others'][:,:, 1] >= crop_y,
                                  self.anno['joint_others'][:,:, 1] < 0))
            # print("__________")
            # print(mask.shape)
            # print(self.anno['joint_others'][mask==True].shape)
            self.anno['joint_others'][mask==True,2] = 2
        # print(self.anno['joint_self'])
        # print(self.anno['joint_others'])
        # visualize_body_part(self.img, self.anno['joint_self'])
        # visualize_body_part(self.img, self.anno['joint_others'])
        # visualize_img(self.mask_miss_crop)
        # visualize_img(self.mask_all_crop)

    def __remove_illegal_joint(self):
        crop_x = int(self.transform['crop_size_x'])
        crop_y = int(self.transform['crop_size_y'])
        mask = np.logical_or.reduce((self.anno['joint_self'][:, 0] >= crop_x,
                                     self.anno['joint_self'][:, 0] < 0,
                                     self.anno['joint_self'][:, 1] >= crop_y,
                                     self.anno['joint_self'][:, 1] < 0))
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        self.anno['joint_self'][mask == True, :] = (1,1,2)
        if (self.anno['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((self.anno['joint_others'][:, :, 0] >= crop_x,
                                         self.anno['joint_others'][:, :, 0] < 0,
                                         self.anno['joint_others'][:, :, 1] >= crop_y,
                                         self.anno['joint_others'][:, :, 1] < 0))
            self.anno['joint_others'][mask == True, :] = (1,1,2)
    def aug_flip(self):
        mode = self.transform['mode']
        num_other_people = self.anno['numOtherPeople']
        dice = random.random()
        doflip = (dice <= self.transform['flip_prob'])

        if(doflip):
            self.img = self.img.copy()
            cv.flip(src=self.img,flipCode=1,dst=self.img)
            w = self.img.shape[1]
            if(mode > 4):
                self.mask_miss = self.mask_miss.copy()
                cv.flip(src=self.mask_miss, flipCode=1, dst=self.mask_miss)
            if(mode > 5):
                self.mask_all = self.mask_all.copy()
                cv.flip(src=self.mask_all,flipCode=1,dst=self.mask_all)

            self.anno['objpos'][0] = w-1-self.anno['objpos'][0]
            self.anno['joint_self'][:,0] = w-1-self.anno['joint_self'][:,0]
            if(num_other_people!=0):
                self.anno['objpos_other'][:,0] = w - 1 -self.anno['objpos_other'][:,0]
                self.anno['joint_others'][:,:,0] = w - 1 - self.anno['joint_others'][:,:,0]

            self.__swapLeftRight()
            self.aug_params['doflip'] = True
        else:
            self.aug_params['doflip'] = False

    def __swapLeftRight(self):

        num_parts = self.transform['np']
        num_other_people = self.anno['numOtherPeople']

        if(num_parts == 56):
            right = [3,4,5, 9,10,11,15,17]
            left = [6,7,8,12,13,14,16,18]
            right = [right[i] - 1 for i in range(len(right))]
            left = [left[i] - 1 for i in range(len(left))]
            temp = self.anno['joint_self'][right,:]
            self.anno['joint_self'][right,:] = self.anno['joint_self'][left,:]
            self.anno['joint_self'][left,:] = temp
            if(num_other_people!=0):
                temp = self.anno['joint_others'][:,right,:]
                self.anno['joint_others'][:,right,:] = self.anno['joint_others'][:,left,:]
                self.anno['joint_others'][:, left, :] = temp

    def visualize(self):
        visualize_body_part(self.img,self.anno['joint_self'])
        # visualize_img(self.mask_miss)
        # visualize_img(self.mask_all)
        visualize_body_part(self.img, self.anno['joint_others'])

    def set_ground_truth(self):
        self.__remove_illegal_joint()
        # print("+++++++++++++++++++++++++")
        # print(self.anno['joint_self'])
        # print(self.anno['joint_others'])
        stride = self.transform['stride']
        mode = self.transform['mode']
        crop_size_y = self.transform['crop_size_y']
        crop_size_x = self.transform['crop_size_x']
        num_parts = self.transform['np']
        nop  = self.anno['numOtherPeople']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        channels = (num_parts+1)*2
        # print grid_x,grid_y,channels
        # the shape of ground-truth 用0初始化gt
        self.gt = np.zeros((grid_y,grid_x,channels))

        # set the channels 0~56
        if(mode > 4):
            self.mask_miss = cv.resize(self.mask_miss, (0, 0), fx=1.0/stride, fy=1.0/stride, interpolation=cv.INTER_CUBIC)
            # gt 0~56 为0 或 1
            self.gt[:,:,:num_parts+1] = np.repeat(self.mask_miss[:,:,np.newaxis],num_parts+1,axis=2)/255
            # print(self.gt[:,:,0])
        if(mode > 5):
            self.mask_all = cv.resize(self.mask_all, (0, 0), fx=1.0/stride, fy=1.0/stride, interpolation=cv.INTER_CUBIC)
            self.gt[:,:,56] = 1
            self.gt[:,:,2*num_parts+1] = self.mask_all/255
        # set the channels 57~113
        if(num_parts == 56):
            # confidance maps for body parts
            for i in range(18):
                if(self.anno['joint_self'][i,2] <= 1):
                    center = self.anno['joint_self'][i,:2]
                    self.gt[:,:,num_parts+1+i+39] = self.__putGaussianMaps(center,self.gt[:,:,num_parts+1+i+39])
                for j in range(nop):
                    if(self.anno['joint_others'][j,i,2]<=1):
                        center = self.anno['joint_others'][j,i, :2]
                        self.gt[:,:,num_parts+1+i+39] = self.__putGaussianMaps(center,self.gt[:,:,num_parts+1+i])
            # pafs
            mid_1 = self.mid_1
            mid_2 = self.mid_2
            # mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
            # mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
            thre = 1
            for i in range(19):
                # limb的两个端点必须可视化
                if(self.anno['joint_self'][mid_1[i]-1,2]<=1 and self.anno['joint_self'][mid_2[i]-1,2]<=1):
                    centerA = self.anno['joint_self'][mid_1[i]-1,:2]
                    centerB = self.anno['joint_self'][mid_2[i]-1,:2]
                    count = np.zeros((grid_y,grid_x),dtype=np.uint32)
                    vec_map = self.gt[:,:,num_parts+1+2*i:num_parts+3+2*i]
                    self.gt[:,:,num_parts+1+2*i:num_parts+3+2*i],count= self.__putVecMaps(centerA=centerA,
                                                                       centerB=centerB,
                                                                       accumulate_vec_map=vec_map,
                                                                       count=count)
                    for j in range(nop):
                        # limb的两个端点必须可视化
                        if(self.anno['joint_others'][j,mid_1[i]-1,2]<=1
                           and self.anno['joint_others'][j,mid_2[i]-1,2]<=1):
                            centerA = self.anno['joint_others'][j,mid_1[i] - 1, :2]
                            centerB = self.anno['joint_others'][j,mid_2[i] - 1, :2]
                            vec_map = self.gt[:, :, num_parts + 1 + 2 * i:num_parts + 3 + 2 * i]
                            self.gt[:, :, num_parts + 1 + 2 * i:num_parts + 3 + 2 * i], count = self.__putVecMaps(centerA=centerA,
                                                                                                    centerB=centerB,
                                                                                                    accumulate_vec_map=vec_map,
                                                                                                    count=count)
    def __putGaussianMaps(self,center,accumulate_confid_map):
        # print('__putGaussianMaps center  ',center)
        crop_size_y = self.transform['crop_size_y']
        crop_size_x = self.transform['crop_size_x']
        stride = self.transform['stride']
        sigma = self.transform['sigma']

        grid_y = crop_size_y/stride
        grid_x = crop_size_x/stride
        start  = stride/2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0])**2 + (yy - center[1])**2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask,cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map>1.0] = 1.0
        return accumulate_confid_map

    def __putVecMaps(self,centerA,centerB,accumulate_vec_map,count):
        # print('__putVecMaps center  ',centerA,centerB)

        stride = self.transform['stride']
        crop_size_y = self.transform['crop_size_y']
        crop_size_x = self.transform['crop_size_x']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        thre = 1 # 用以定义肢干的宽度
        centerB = centerB / stride
        centerA = centerA / stride
        limb_vec = centerB - centerA
        limb_vec_unit = limb_vec / np.linalg.norm(limb_vec)
        # print('the unit vector of the limb =========>   ',limb_vec_unit)

        min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
        max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
        min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
        max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)
        # print(min_x,max_x,min_y,max_y)
        # print(centerA,centerB)
        # exit()
        range_x = range(min_x,max_x,1)
        range_y = range(min_y,max_y,1)

        xx, yy = np.meshgrid(range_x, range_y)
        ba_x = xx - centerA[0] # 点(x,y)到centerA的向量
        ba_y = yy - centerA[1]
        limb_width = np.abs(ba_x*limb_vec_unit[1] - ba_y*limb_vec_unit[0])
        mask = limb_width < thre # mask为2D
        # print('mask shape =====>  ',mask.shape)
        vec_map = np.copy(accumulate_vec_map)*0.0
        vec_map[yy,xx] = np.repeat(mask[:,:,np.newaxis],2,axis=2)
        vec_map[yy,xx] *= limb_vec_unit[np.newaxis,np.newaxis,:]
        # print(vec_map)
        mask = np.logical_or.reduce((vec_map[:,:,0]>0,vec_map[:,:,1]>0))
        # count = np.repeat(count[:,:,np.newaxis],2,axis=2)
        accumulate_vec_map =  np.multiply(accumulate_vec_map,count[:,:,np.newaxis])
        accumulate_vec_map += vec_map
        count[mask == True] += 1
        mask = count == 0
        count[mask==True] = 1
        accumulate_vec_map = np.divide(accumulate_vec_map,count[:,:,np.newaxis])
        count[mask==True] = 0
        # print("**************  ",accumulate_vec_map.shape)
        return accumulate_vec_map,count

    def get_sample_label(self):
        temp = self.img.astype(float)
        sample = (self.img - 128)/256.0
        label = self.gt
        return (sample,label)

    def visualize_heat_maps(self):
        heat_maps = self.gt[:,:,95:]
        for i in range(18):
            self.visualize_heat_map(heat_maps[:,:,i])

    def visualize_heat_map(self,heatmap):
        stride = self.transform['stride']
        heatmap = cv.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        # visualization
        f,axarr = plt.subplots(1,3)
        axarr[0].imshow(self.img[:, :, [2, 1, 0]])
        axarr[1].imshow(heatmap, alpha=.5)
        axarr[2].imshow(self.img[:, :, [2, 1, 0]])
        axarr[2].imshow(heatmap, alpha=.5)
        plt.show()

    def visualize_pafs_single_figure(self):

        stride = self.transform['stride']
        pafs = self.gt[:, :, 57:95]
        idx = 0
        plt.figure()
        for i in range(19):
            U = pafs[:, :, idx]
            V = pafs[:, :, idx + 1]
            U = cv.resize(U, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
            V = cv.resize(V, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
            # self.visualize_paf(U, V)
            U = U * -1
            # V = vector_f_y[:, :, 17]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')
            M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            # 1

            plt.imshow(self.img[:, :, [2, 1, 0]], alpha=.5)
            s = 5
            Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                           scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')
            idx += 2
        # fig = plt.gcf()
        # fig.set_size_inches(20, 20)
        plt.show()

    def visualize_pafs(self):
        stride = self.transform['stride']
        pafs = self.gt[:,:,57:95]
        idx = 0
        for i in range(19):
            U = pafs[:,:,idx]
            V = pafs[:,:,idx+1]
            U = cv.resize(U,(0,0),fx=stride,fy=stride,interpolation=cv.INTER_CUBIC)
            V = cv.resize(V,(0,0),fx=stride,fy=stride,interpolation=cv.INTER_CUBIC)
            self.visualize_paf(U,V)
            idx +=2

    def visualize_paf(self,U,V):

        U = U * -1
        # V = vector_f_y[:, :, 17]
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
        M = np.zeros(U.shape, dtype='bool')
        M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
        U = ma.masked_array(U, mask=M)
        V = ma.masked_array(V, mask=M)

        # 1
        plt.figure()
        plt.imshow(self.img[:, :, [2, 1, 0]], alpha=.5)
        s = 5
        Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                       scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.show()









