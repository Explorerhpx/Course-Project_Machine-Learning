from processing_data import *
import cv2
import pickle
from sklearn.cross_validation import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt


def ft_img(image):
    """calculating the ** feature-images of every painting"""
    ft_img = []
    for k in range(24):
        res = cv2.filter2D(image, -1, gabor_filter[k])
        ft_img += [res]
        print(k)
        del res
    return ft_img

def stat_1(image):
    """renturn the mean of the image"""
    return image.mean()


def stat_2(image):
    """return the std of the image"""
    return np.std(image)


def stat_3(image):
    """reutrn the precent of the tail"""
    n, m = image.shape
    mean = stat_1(image)
    std = stat_2(image)
    count = sum(sum(abs(image - mean) > std))
    return count / (m * n)

def stat_4(image):
    """return the energy value of the image"""
    return sum(sum(image**2))

def gabor_fn(sigma, alpha, Lambda, psi, gamma):
    """In our problem, sigma is std of Gaussian part,
      alpha is orientation, Lambda is wave length, psi=0, gamma=1"""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    max = 30
    min = -max
    (y, x) = np.meshgrid(np.arange(min, max + 1), np.arange(min, max + 1))

    # Rotation
    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)

    gb = np.exp(-.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_alpha + psi)
    return gb

def feature_set(S):
    """extracting 54 feature for every painting in a whole set,
    S = N,T,D"""
    for k in range(len(S)):
        image = S[k]
        ft = ft_img(image)
        single_S = []
        for loop in range(len(ft)):
            # save the 96 features
            st_1 = stat_1(ft[loop])
            st_2 = stat_2(ft[loop])
            st_3 = stat_3(ft[loop])
            st_4 = stat_4(ft[loop])
            single_S += [st_1, st_2, st_3,st_4]
        # index = [2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27] # for T
        # index = [1,7,10,20,23,25,26] # for D
        index = [11, 12, 13, 14, 15, 16, 17, 18, 19]  # for N
        pickle.dump(single_S, open('data_4_parts_gabor/N_' + str(index[k//4]) + '_'+str(k%4+1) +'.p', 'wb'))

if __name__ == '__main__':
    '''loading the gray-scale and edge-truncated painting data'''
    raphael_D = pickle.load(open('data_4_parts_gabor/truncated_edge_D.p', 'rb'))
    raphael_T = pickle.load(open('data_4_parts_gabor/truncated_edge_T.p', 'rb'))
    raphael_N = pickle.load(open('data_4_parts_gabor/truncated_edge_N.p', 'rb'))

    # """setting the 24 gabor filters"""
    # sigma = 10
    # psi = 0
    # gamma = 1
    # gabor_filter = []
    # for i in range(6):
    #     for j in range(4):
    #         alpha = i * np.pi/6
    #         Lambda = 5*(j+1)
    #         gabor_filter += [np.array(gabor_fn(sigma, alpha, Lambda, psi, gamma))]
    #
    # """extracting 54 feature for every painting set: N,T,D"""
    # feature_set(raphael_T)
    # feature_set(raphael_D)
    # feature_set(raphael_N)

    tight_frame_data = []
    path = 'data_4_parts_gabor'
    Files_name = os.listdir(path)
    for file in Files_name:  # file是文件名（有后缀名）
        if not os.path.isdir(file):
            if (file != '.DS_Store'):
                with open(path + "/" + file, 'rb') as load_file:
                    print(file)
                    a=pickle.load(load_file)
                    tight_frame_data += [a]

    D = tight_frame_data[0:len(raphael_D)]
    N = tight_frame_data[len(raphael_D):len(raphael_N)+len(raphael_D)]
    T = tight_frame_data[len(raphael_N)+len(raphael_D):len(raphael_N)+len(raphael_D)+len(raphael_T)]

    pickle.dump(np.array(T), open('data_4_parts_gabor/tight_frame_T.p', 'wb'))
    pickle.dump(np.array(N), open('data_4_parts_gabor/tight_frame_N.p', 'wb'))
    pickle.dump(np.array(D), open('data_4_parts_gabor/tight_frame_D.p', 'wb'))
