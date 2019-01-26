load digits.mat
[n,d] = size(X);
nLabels = max(y);  %lbel的种类数
yExpanded = linearInd2Binary(y,nLabels);   %把y转换为-1/1表示的二维向量
t = size(Xvalid,1);  %t: Xvalid的数据量
t2 = size(Xtest,1);  %t2: Xtest的数据量

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);  %数据规范化--》为N(0,1)
X = [ones(n,1) X];  %加入bias
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
% 神经网络结构
nHidden = [400];  %隐层的神经元数目，长度代表网络的隐层数目

% Count number of parameters and initialize weights 'w'
% 统计参数个数

nParams = d*nHidden(1);  %输入层次的参数个数; d:输入数据的维数
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);  %逐层加上每一隐层的参数个数
end
nParams = nParams+nHidden(end)*nLabels;   %加上输出层的参数个数
w = randn(nParams,1);  %随机初始化参数

T1 = clock;
%% Train with stochastic gradient
% 随机梯度下降法进行训练
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

f0 = 1; %连续4次取样
f1 = 1; %标记未取样
f2 = 1;
w_best = w;

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0  %输出训练过程
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    
    if mod(iter-1,round(maxIter/100)) == 0  %取样
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        error = sum(yhat~=yvalid)/t;
        if error < f0 %结果变好
            f0 = error;
            w_best = w;
            f1 = 1; f2 = 1;
        elseif f1    %第一次取样
            f1 = 0;
        elseif f2
            f2 = 0; %第二次取样
        else
            w = w_best;
            break; %连续3次取样结果均未变好 
        end
    end
    
    i = ceil(rand*n);  %随机梯度下降
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
end

T2 = clock;
etime(T2,T1)

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);