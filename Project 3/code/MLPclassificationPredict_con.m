function [y] = MLPclassificationPredict_con(w,lkernal,X,nHidden,nLabels)
[nInstances,nv] = size(X);
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%进行卷积运算
Kernal = w(1:lkernal^2); %构造卷积核
w = w(1+lkernal^2:end);
Kernal = reshape(Kernal,[lkernal,lkernal]);

T = [];

for i = 1:nInstances
    A = reshape(X(i,:),[sqrt(nv),sqrt(nv)]);  %重构图片
    B = conv2(A,Kernal,'valid');  %进行卷积
    c = reshape(B,[1,numel(B)]);  %重构卷积结果
    T = [T;c];
end

X = T;
[nInstances,nVars] = size(X);

%!!!!!!!!!!!!!!!!!!!!!!!!!
% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));  %!!!!!!!!!!!!!!!
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);   % 找出最大值
%y = binary2LinearInd(y);
