load digits.mat
[n,d] = size(X);
nLabels = max(y);  %lbel��������
yExpanded = linearInd2Binary(y,nLabels);   %��yת��Ϊ-1/1��ʾ�Ķ�ά����
t = size(Xvalid,1);  %t: Xvalid��������
t2 = size(Xtest,1);  %t2: Xtest��������

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);  %���ݹ淶��--��ΪN(0,1)
X = [ones(n,1) X];  %����bias
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
% ������ṹ
nHidden = [400];  %�������Ԫ��Ŀ�����ȴ��������������Ŀ

% Count number of parameters and initialize weights 'w'
% ͳ�Ʋ�������

nParams = d*nHidden(1);  %�����εĲ�������; d:�������ݵ�ά��
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);  %������ÿһ����Ĳ�������
end
nParams = nParams+nHidden(end)*nLabels;   %���������Ĳ�������
w = randn(nParams,1);  %�����ʼ������

T1 = clock;
%% Train with stochastic gradient
% ����ݶ��½�������ѵ��
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

f0 = 1; %����4��ȡ��
f1 = 1; %���δȡ��
f2 = 1;
w_best = w;

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0  %���ѵ������
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    
    if mod(iter-1,round(maxIter/100)) == 0  %ȡ��
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        error = sum(yhat~=yvalid)/t;
        if error < f0 %������
            f0 = error;
            w_best = w;
            f1 = 1; f2 = 1;
        elseif f1    %��һ��ȡ��
            f1 = 0;
        elseif f2
            f2 = 0; %�ڶ���ȡ��
        else
            w = w_best;
            break; %����3��ȡ�������δ��� 
        end
    end
    
    i = ceil(rand*n);  %����ݶ��½�
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
end

T2 = clock;
etime(T2,T1)

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);