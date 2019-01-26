%对加入卷积层的神经网络进行测试

load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
% X = [ones(n,1) X];  %去掉输入层的bias   !!!!!!!!!!!!!!

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
%Xvalid = [ones(t,1) Xvalid];   !!!!!!!!!!!!!!!!
Xtest = standardizeCols(Xtest,mu,sigma);
%Xtest = [ones(t2,1) Xtest];   !!!!!!!!!!!!!!!!

% Choose network structure
% 神经网络结构
nHidden = [400];
lkernal = 5;  %卷积核的大小  !!!!!!!!!!!!

% Count number of parameters and initialize weights 'w'
% 统计参数个数
D = (sqrt(d)+1-lkernal)^2;  %卷积后的输出大小
nParams = D*nHidden(1);  %输入层次的参数个数; d:输入数据的维数
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);  %逐层加上每一隐层的参数个数
end
nParams = nParams+nHidden(end)*nLabels + lkernal^2;   %加上输出层的参数个数
w = randn(nParams,1);  %随机初始化参数

T1 = clock;
%% Train with stochastic gradient
% 随机梯度下降法进行训练
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss_con(w,lkernal,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0  %输出训练过程
        yhat = MLPclassificationPredict_con(w,lkernal,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    i = ceil(rand*n);  %随机梯度下降
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
    % w = fine_tuning(w,X(i,:),yExpanded(i,:),nHidden,nLabels); %Finetuinig
    
end

T2 = clock;
etime(T2,T1)

% Evaluate test error
yhat = MLPclassificationPredict_con(w,lkernal,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);