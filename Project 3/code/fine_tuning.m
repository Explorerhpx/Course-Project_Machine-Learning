function [ W ] = fine_tuning( w,X,y,nHidden,nLabels )
% finetuning the weights

[nInstances,nVars] = size(X);
% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
    hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
    offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);


ip{1} = X*inputWeights;
fp{1} = tanh(ip{1});    % activate function 
for h = 2:length(nHidden)
    ip{h} = fp{h-1}*hiddenWeights{h-1};
    fp{h} = tanh(ip{h});
end

XX = fp{end}; % the input of the last layer
YY = y;
Maxiter = 100;  % iterations
step_size = 0.001; % step size

% gradient descent
for ite = 1 : Maxiter
    step_size = step_size / ite; % change step size
    delta = XX' * (XX * outputWeights - YY);
    outputWeights = outputWeights + step_size * delta; % gradient descent
end

% Form new weights
g = w;
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
    offset = offset+nHidden(h-1)*nHidden(h);
end
g(offset+1:offset+nHidden(end)*nLabels) = outputWeights(:);
W = g;
end

