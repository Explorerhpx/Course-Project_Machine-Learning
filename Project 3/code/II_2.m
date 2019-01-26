load digits.mat
[n,d] = size(X);
nLabels = max(y); 
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [500];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels; 
w = randn(nParams,1);

%% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)MLPclassificationLoss_IIII_1(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

pre_error = 1;  % Pre_error, default: 1

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        %fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        fprintf('%f\n',sum(yhat~=yvalid)/t);
    end
    
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        current_error = sum(yhat~=yvalid)/t;
        if current_error >= pre_error
            stepSize = stepSize/2;
        else
            pre_error = current_error;
        end
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g; 
    if (iter>maxIter/2)
        w = fine_tuning(w,X(i,:),yExpanded(i,:),nHidden,nLabels); %Finetuinig
    end
end

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);