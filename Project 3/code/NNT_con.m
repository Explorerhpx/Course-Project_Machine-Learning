%�Լ����������������в���

load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
% X = [ones(n,1) X];  %ȥ��������bias   !!!!!!!!!!!!!!

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
%Xvalid = [ones(t,1) Xvalid];   !!!!!!!!!!!!!!!!
Xtest = standardizeCols(Xtest,mu,sigma);
%Xtest = [ones(t2,1) Xtest];   !!!!!!!!!!!!!!!!

% Choose network structure
% ������ṹ
nHidden = [400];
lkernal = 5;  %����˵Ĵ�С  !!!!!!!!!!!!

% Count number of parameters and initialize weights 'w'
% ͳ�Ʋ�������
D = (sqrt(d)+1-lkernal)^2;  %�����������С
nParams = D*nHidden(1);  %�����εĲ�������; d:�������ݵ�ά��
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);  %������ÿһ����Ĳ�������
end
nParams = nParams+nHidden(end)*nLabels + lkernal^2;   %���������Ĳ�������
w = randn(nParams,1);  %�����ʼ������

T1 = clock;
%% Train with stochastic gradient
% ����ݶ��½�������ѵ��
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss_con(w,lkernal,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0  %���ѵ������
        yhat = MLPclassificationPredict_con(w,lkernal,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    i = ceil(rand*n);  %����ݶ��½�
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
    % w = fine_tuning(w,X(i,:),yExpanded(i,:),nHidden,nLabels); %Finetuinig
    
end

T2 = clock;
etime(T2,T1)

% Evaluate test error
yhat = MLPclassificationPredict_con(w,lkernal,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);