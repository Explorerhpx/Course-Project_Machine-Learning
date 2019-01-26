# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle


def tran_grey(image):
    '''transforming image to grayscale 通过将图片三通道加权平均'''
    image_1 = np.array(image)
    image_2 = image_1[:,:,0]*299+image_1[:,:,1]*587+image_1[:,:,2]*114
    image_2 = image_2//1000 #整除方便起见
    return image_2

def truncate(image_1):
    """cutting the edges
    n  is  the num of pixel cutting from each edge"""
    n = 60 #100
    i,j = image_1.shape
    image_2 = image_1[n:i-n,n:j-n]
    return image_2

def cut4(image):
    """cut into 4 parts"""
    i, j = image.shape
    a1 = image[:i // 2, :j // 2]
    a2 = image[i // 2:, :j // 2]
    a3 = image[:i // 2, j // 2:]
    a4 = image[i // 2:, j // 2:]
    return  a1, a2, a3, a4
if __name__=='__main__':
    '''loading data manually,since they are of differnet forms(tif,tiff,jpg)
     D means the disputed data
     N means non-raphael data
     T mean Raphael's data'''

    raphael_D = []
    raphael_1 = mpimg.imread('1.D.tif')
    raphael_7 = mpimg.imread('7.D.tiff')
    raphael_10 = mpimg.imread('10.D.tif')
    raphael_20 = mpimg.imread('20.D.tif')
    raphael_23 = mpimg.imread('23.D.tif')
    raphael_25 = mpimg.imread('25.D.tif')
    raphael_26 = mpimg.imread('26.D.tif')
    raphael_D = raphael_D + [raphael_1,raphael_7,raphael_10,raphael_20,\
                         raphael_23,raphael_25,raphael_26]
    print('loading raphael_D done!')

    raphael_T = []
    raphael_2 = mpimg.imread('2.T.tif')
    raphael_3 = mpimg.imread('3.T.tif')
    raphael_4 = mpimg.imread('4.T.tiff')
    raphael_5 = mpimg.imread('5.T.tiff')
    raphael_6 = mpimg.imread('6.T.tiff')
    raphael_8 = mpimg.imread('8.T.tif')
    raphael_9 = mpimg.imread('9.T.tif')
    raphael_21 = mpimg.imread('21.T.jpg')
    raphael_22 = mpimg.imread('22.T.jpg')
    raphael_24 = mpimg.imread('24.T.tif')
    raphael_27 = mpimg.imread('27.T.tiff')
    raphael_T = raphael_T + [raphael_2,raphael_3,raphael_4,raphael_5,\
                         raphael_6,raphael_8,raphael_9,raphael_21,\
                         raphael_22,raphael_24,raphael_27]
    print('loading raphael_T done!')

    raphael_N = []
    raphael_11 = mpimg.imread('11.N.jpg')
    raphael_12 = mpimg.imread('12.N.jpg')
    raphael_13 = mpimg.imread('13.N.jpg')
    raphael_14 = mpimg.imread('14.N.jpg')
    raphael_15 = mpimg.imread('15.N.jpg')
    raphael_16 = mpimg.imread('16.N.jpg')
    raphael_17 = mpimg.imread('17.N.jpg')
    raphael_18 = mpimg.imread('18.N.jpg')
    raphael_19 = mpimg.imread('19.N.jpg')
    raphael_N = raphael_N + [raphael_11,raphael_12,raphael_13,raphael_14,\
                         raphael_15,raphael_16,raphael_17,raphael_18,\
                         raphael_19]
    print('loading raphael_N done!')

    ''' transforming painting into grey-scale'''
    raphael_D_2 = []
    for i in raphael_D:
        raphael_D_2 += [tran_grey(i)]
        
    raphael_N_2 = []
    for j in raphael_N:
        raphael_N_2 += [tran_grey(j)]
        
    raphael_T_2 = []
    for k in raphael_T:
        raphael_T_2 += [tran_grey(k)]
    print('transformation done!')

    ''' truncating the edges of the gray scale paintings'''
    raphael_D_3 = []
    for i in raphael_D_2:
        raphael_D_3 += [truncate(i)]
    
    raphael_N_3 = []
    for j in raphael_N_2:
        raphael_N_3 += [truncate(j)]

    raphael_T_3 = []
    for k in raphael_T_2:
        raphael_T_3 += [truncate(k)]
    print('truncation done!')

    '''17.N need special truncation'''
    image = raphael_N_3[6]
    i,j = image.shape
    image_2 = image[500:i-500,100:j-100]
    raphael_N_3[6] = image_2

    '''split into 4 parts'''
    raphael_D_4 = []
    for i in raphael_D_3:
        a1,a2,a3,a4 = cut4(i)
        raphael_D_4 += [a1,a2,a3,a4]

    raphael_N_4 = []
    for j in raphael_N_3:
        a1, a2, a3, a4 = cut4(j)
        raphael_N_4 += [a1,a2,a3,a4]

    raphael_T_4 = []
    for k in raphael_T_3:
        a1, a2, a3, a4 = cut4(k)
        raphael_T_4 += [a1,a2,a3,a4]
    print('cut done!')

    # saving the processed data:
    pickle.dump(raphael_N_4,open('data_4_parts/truncated_edge_N.p','wb'))
    pickle.dump(raphael_T_4,open('data_4_parts/truncated_edge_T.p','wb'))
    pickle.dump(raphael_D_4,open('data_4_parts/truncated_edge_D.p', 'wb'))

    



