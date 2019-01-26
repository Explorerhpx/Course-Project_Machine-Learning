function [y] = MLPclassificationPredict_VI(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

alpha = 2;  %����bias

% Form Weights
% ����Ȩֵ
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
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
    fp{1}(end) = alpha;  %!!!!!!!!!!!!!
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
        fp{h}(end) = alpha;  %!!!!!!!!!!
    end
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);   % �ҳ����ֵ
%y = binary2LinearInd(y);
