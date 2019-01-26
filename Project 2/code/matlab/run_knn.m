function [valid_labels] = run_knn(k, train_data, train_labels, valid_data)
% knn_predict: Uses the supplied training inputs and labels to make
%              predictions for validation data using the K-nearest neighbours
%              algorithm.
%
% Note: N_TRAIN is the number of training examples,
%       N_VALID is the number of validation examples, 
%       and M is the number of features per example.
%
% Inputs:
%   k:            The number of neighbours to use for classification 
%                 of a validation example.
%   train_data:   The N_TRAIN x M matrix of training
%                 data.
%   train_labels: The N_TRAIN x 1 vector of training labels
%                 corresponding to the examples in train_data 
%                 (must be binary).
%   valid_data:    The N_VALID x M matrix of data to
%                 predict classes for.
%
% Outputs:
%   valid_labels: The N_VALID x 1 vector of predicted labels for the validation data.
%

error(nargchk(4,4,nargin));  %���������������Ϊ4��

dist = l2_distance(valid_data', train_data');  %�����valid_data��train_data��ľ�����
[sorted_dist, nearest] = sort(dist,2);   %ÿһ�ж������н�������nearest����洢ÿ��valid�������k��train��

nearest = nearest(:,1:k);    %ȡ��nearst����ĵ�1��k��
valid_labels = train_labels(nearest);

% note this only works for binary labels
valid_labels = mean(valid_labels,2) >= 0.5;   %ʹ��ƽ��ֵ��
