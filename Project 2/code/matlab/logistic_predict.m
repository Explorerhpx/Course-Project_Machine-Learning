function [y] = logistic_predict(weights, data)
%    Compute the probabilities predicted by the logistic classifier.
%
%    Note: N is the number of examples and 
%          M is the number of features per example.
%
%    Inputs:
%        weights:    (M+1) x 1 vector of weights, where the last element
%                    corresponds to the bias (intercepts).
%        data:       N x M data matrix where each row corresponds 
%                    to one data point.
%    Outputs:
%        y:          :N x 1 vector of probabilities. This is the output of the classifier.

%TODO: finish this function
[N,M] = size(data);
data = [data'; ones(1,N)]'; %对data进行扩维处理
y = zeros(N,1);
for i = 1: N
    tem = 1/(1+exp(- data(i,:) * weights));  %预测
    y(i) = tem;
end
end
