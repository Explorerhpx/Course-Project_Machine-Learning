
from matplotlib import pyplot
from processing_data import *
import cv2
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
import os



def ft_img(image):
    """calculating the ** feature-images of every painting"""
    ft_img = []
    for k in range(24):
        res = cv2.filter2D(image, -1, tau[k])
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


def feature_set(S):
    """extracting 54 feature for every painting in a whole set,
    S = N,T,D"""
    for k in range(len(S)):
        image = S[k]
        ft = ft_img(image)
        single_S = []
        for loop in range(len(ft)):
            st_1 = stat_1(ft[loop])
            st_2 = stat_2(ft[loop])
            st_3 = stat_3(ft[loop])
            single_S += [st_1, st_2, st_3]
        # save the 54 features

        # index = [2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27] # for T
        # index = [1,7,10,20,23,25,26] # for D
        index = [11, 12, 13, 14, 15, 16, 17, 18, 19]  # for N
        pickle.dump(single_S, open('data_4_parts/N_' + str(index[k//4]) + '_'+str(k%4+1) +'.p', 'wb'))
        # np.save('data/_' + str(index[k]),single_S)


if __name__ == '__main__':
    '''loading the gray-scale and edge-truncated painting data'''
    raphael_D = pickle.load(open('data_4_parts/truncated_edge_D.p', 'rb'))
    raphael_T = pickle.load(open('data_4_parts/truncated_edge_T.p', 'rb'))
    raphael_N = pickle.load(open('data_4_parts/truncated_edge_N.p', 'rb'))



    '''setting the 18 filters'''
    # tau = []
    # tau_0 = np.array([[1,2,1],      [2,4,2],    [1,2,1]])/16
    # tau_1 = np.array([[1,0,-1],     [2,0,-2],   [1,0,-1]])/16
    # tau_2 = np.array([[1,2,1],      [0,0,0],    [-1,-2,-1]])/16
    # tau_3 = np.array([[1,1,0],      [1,0,-1],   [0,-1,-1]])*(2**0.5)/16
    # tau_4 = np.array([[0,1,1],      [-1,0,1],   [-1,-1,0]])*(2**0.5)/16
    # tau_5 = np.array([[1,0,-1],     [0,0,0],    [-1,0,1]])*(7**0.5)/24
    # tau_6 = np.array([[-1,2,-1],    [-2,4,-2],  [-1,2,-1]])/48
    # tau_7 = np.array([[-1,2,-1],    [2,4,2],    [-1,-2,-1]])/48
    # tau_8 = np.array([[0,0,-1],     [0,2,0],    [-1,0,0]])/12
    # tau_9 = np.array([[-1,0,0],     [0,2,0],    [0,0,-1]])/12
    # tau_10 = np.array([[0,1,0],     [-1,0,-1],  [0,1,0]])*(2**0.5)/12
    # tau_11 = np.array([[-1,0,1],    [2,0,-2],   [-1,0,1]])*(2**0.5)/16
    # tau_12 = np.array([[-1,2,-1],   [0,0,0],    [1,-2,1]])*(2**0.5)/16
    # tau_13 = np.array([[1,-2,1],    [-2,4,-2],  [1,-2,1]])/48
    # tau_14 = np.array([[0,0,0],     [-1,2,-1],  [0,0,0]])*(2**0.5)/12
    # tau_15 = np.array([[-1,2,-1],   [0,0,0],    [-1,2,-1]])*(2**0.5)/24
    # tau_16 = np.array([[0,-1,0],    [0,2,0],    [0,-1,0]])*(2**0.5)/12
    # tau_17 = np.array([[-1,0,-1],   [2,0,2],    [-1,0,-1]])*(2**0.5)/24
    # tau += [tau_0,tau_1,tau_2,tau_3,tau_4,tau_5,tau_6,tau_7,tau_8,tau_9,\
    #         tau_10,tau_11,tau_12,tau_13,tau_14,tau_15,tau_16,tau_17]
    # pickle.dump(tau,open('data/tau.p','wb'))
    # tau = pickle.load(open('data/tau.p', 'rb'))

    """extracting 54 feature for every painting set: N,T,D"""
    # feature_set(raphael_T)
    # feature_set(raphael_D)
    # feature_set(raphael_N)



    # # devide training, test set
    # fake = pickle.load(open('tight_frame_N.p', 'rb'))
    # true = pickle.load(open('tight_frame_T.p', 'rb'))
    # set = np.column_stack((fake.T, true.T)).T
    # Label = np.zeros((fake.shape[0] + true.shape[0]))
    # Label[len(fake):] += 1  # label of true masterpiece is 1
    #
    # X_train, X_test, y_train, y_test = train_test_split(set, Label, test_size=0.25)
    # train_set_file = open('Train_set.pkl', 'wb')
    # train_label_file = open('Train_label.pkl', 'wb')
    # test_set_file = open('Test_set.pkl', 'wb')
    # test_label_file = open('Test_label.pkl', 'wb')
    # pickle.dump(X_train, train_set_file)
    # pickle.dump(X_test, test_set_file)
    # pickle.dump(y_train, train_label_file)
    # pickle.dump(y_test, test_label_file)

    # tight_frame_data = []
    # path = 'data_4_parts'
    # Files_name = os.listdir(path)
    # for file in Files_name:  # file是文件名（有后缀名）
    #     if not os.path.isdir(file):
    #         if (file != '.DS_Store'):
    #             with open(path + "/" + file, 'rb') as load_file:
    #                 print(file)
    #                 a=pickle.load(load_file)
    #                 tight_frame_data += [a]
    #
    # D = tight_frame_data[0:len(raphael_D)]
    # N = tight_frame_data[len(raphael_D):len(raphael_N)+len(raphael_D)]
    # T = tight_frame_data[len(raphael_N)+len(raphael_D):len(raphael_N)+len(raphael_D)+len(raphael_T)]
    #
    # pickle.dump(np.array(T), open('data_4_parts/tight_frame_T.p', 'wb'))
    # pickle.dump(np.array(N), open('data_4_parts/tight_frame_N.p', 'wb'))
    # pickle.dump(np.array(D), open('data_4_parts/tight_frame_D.p', 'wb'))




