load digits.mat
[n,d] = size(X);
nLabels = max(y);  %the number of label
yExpanded = linearInd2Binary(y,nLabels);   % transform y into a 2D vectors represented by -1/1
t = size(Xvalid,1);  %t: the data volume of Xvalid
t2 = size(Xtest,1);  %t2: the data volume Xtest

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);  % normalization
X = [ones(n,1) X];  % add bias
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
% the structure of NN
nHidden = [500];  %the number of nurons, len(nHidden) is the number of hidden layers

% Count number of parameters and initialize weights 'w'

nParams = d*nHidden(1);  % the number of parameters belongs to the input layer; d: the dimention of input data
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);  % cumulate the number of parameters layer by layer
end
nParams = nParams+nHidden(end)*nLabels;   % add the number of parameters of output layer
w = randn(nParams,1);  % initialize parameters randomly

T1 = clock;
%% Train with stochastic gradient
% SGD
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss_III(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0  % outpur training process
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);  % SGD
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
    % w = fine_tuning(w,X(i,:),yExpanded(i,:),nHidden,nLabels); %Finetuinig
    
end

T2 = clock;
etime(T2,T1)

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);