# devide training, test set
import numpy as np
import pickle
#import sklearn
from sklearn.cross_validation import train_test_split
# devide training, test set
fake = pickle.load(open('tight_frame_N.p', 'rb'))
true = pickle.load(open('tight_frame_T.p', 'rb'))
set = np.column_stack((fake.T,true.T)).T
Label = np.zeros((fake.shape[0] + true.shape[0]))
Label[len(fake):] += 1 # label of true masterpiece is 1

X_train, X_test, y_train, y_test = train_test_split(set,Label,test_size=0.25)
train_set_file = open('Train_set.pkl', 'wb')
train_label_file = open('Train_label.pkl', 'wb')
test_set_file = open('Test_set.pkl', 'wb')
test_label_file = open('Test_label.pkl', 'wb')
pickle.dump(X_train,train_set_file)
pickle.dump(X_test,test_set_file)
pickle.dump(y_train,train_label_file)
pickle.dump(y_test,test_label_file)