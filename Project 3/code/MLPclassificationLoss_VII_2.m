function [f,g] = MLPclassificationLoss_VII_2(w,X,y,nHidden,nLabels)
% ͨ��ѵ������ѵ��������ۻ�����f �� �ݶ��½�ֵ g

[nInstances,nVars] = size(X);   %nInstances:���Ӹ���,nVars:����ά��

P = 0.8; % Dropout ����

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));  %������weights
hide = diag(binornd(1,P,1,nHidden(1)));  %������һ�������  !!!!!!!!!!!!!!!!!!
inputWeights = inputWeights * hide;  %������ǰһ���weights  !!!!!!!!!!!!!!

offset = nVars*nHidden(1);
for h = 2:length(nHidden)   %ȡ������Weights
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  
  hiddenWeights{h-1} = hide * hiddenWeights{h-1}; %������һ��dropout�뱾���weights   !!!!!!!!!!!!!!!!
  hide = diag(binornd(1,P,1,nHidden(h-1)));   %!!!!!!!!!!!!!
  hiddenWeights{h-1} = hiddenWeights{h-1} * hide ; %��������dropout����һ���weights !!!!!!!!!
  
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);  %�����Ȩ

outputWeights = hide * outputWeights;  %!!!!!!!!!!!!!!!!

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
% ����������
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});    % �����
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end}*outputWeights;
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);  %����
    
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        gOutput = gOutput + repmat(fp{end}',[1,length(err)]) * diag(err);

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            
            backprop = diag(err)*(outputWeights' * diag(sech(ip{end}).^2));
            gHidden{end} = gHidden{end} + repmat(fp{end-1}',[1,size(backprop,1)]) * backprop;

            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
        else  %������
           % Input Weights
           gInput = gInput + repmat(X(i,:)',[1,size(outputWeights',1)]) * (diag(err)*(outputWeights' * diag(sech(ip{end}).^2)));
           
        end
        
    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
