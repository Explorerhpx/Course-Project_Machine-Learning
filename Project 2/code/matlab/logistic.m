function [f, df, y] = logistic(weights, data, targets, hyperparameters) %
% 由当前weights\观测数据：输出-log likelihood, likehood的导数, 预测结果
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value?i.e. negative log likelihood).
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

%计算预测值
y = logistic_predict(weights,data);
[N,M] = size(data);

%计算likehood
data = [data'; ones(1,N)]'; %对data进行扩维处理
Y = zeros(1,N); %存储每一个样本的分类计算结果
for i = 1:N
    tem = - data(i,:) * weights;
    tem = (1 - targets(i))*tem - log(1 + exp(tem));
    Y(i) = -tem;
end
f = sum(Y);

df = zeros(M+1 ,1);
%计算likehood的偏导
for i = 1:N
    tem = - data(i,:) * weights;
    tem = (1 - targets(i)) - exp(tem)/(1+exp(tem));
    df = df + tem*(data(i,:))';
end
end
