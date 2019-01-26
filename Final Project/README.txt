Question:
Identification and Synthesizing of Raphael¡¯s paintings from the forgeries
Member: Pingxuan Huang 15307130283/ Hanwen Wu 15300180043/ Xinzhe Luo 15300180006

Question 1
Step:
Use processing_data.py & train_test_devide.py to preprocess data
Use either train_test_devide.py or tight_frame_feature.py to extract features
Use Feature_model to select features
Use Model.py to build model and do classification
--------------------------------------------------------------------------------------------------------------------------------------
processing_data.py:       Preprocess data: transform RGB images to gray images and so on.
train_test_devide.py:      Devide training/test set
tight_frame_feature.py:  Use tight frame method to extract features
gabor_feature.py:           Use Wavelet method to extract features
Feature_model.py:         Apply cross-one-validation on forward stage wise algorithm to select features
feature_selection.py:      Use stagewise forward method to assist Feature_model.py to select features
Model.py:                       Training and test model
==============================================================================
Question2
Step:
Build 3 folders named as: contentImages\ outputImages\ stuleImages
Put sorce images into corresponding folders
Use synthesis.py to combine the content and style images
-----------------------------------------------------------------------------------------------------------------------------------------
synthesis.py:                 Use CNN to combine content and style images

