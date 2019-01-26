function [f,g] = MLPclassificationLoss_con(w,lkernal,X,y,nHidden,nLabels)
% ͨ��ѵ������ѵ��������ۻ�����f �� �ݶ��½�ֵ g

[nInstances,nv] = size(X);   %nInstances:���Ӹ���,nVars:����ά��

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%���о������
Kernal = w(1:lkernal^2); %��������
w = w(1+lkernal^2:end);
Kernal = reshape(Kernal,[lkernal,lkernal]);

T = [];

for i = 1:nInstances
    A = reshape(X(i,:),[sqrt(nv),sqrt(nv)]);  %�ع�ͼƬ
    B = conv2(A,Kernal,'valid');  %���о��
    c = reshape(B,[1,numel(B)]);  %�ع�������
    T = [T;c];
end
Input = X; %��������ֵ���ڸ��¾����weights
X = T;
[nInstances,nVars] = size(X);

gKernal = zeros(lkernal,lkernal);

%!!!!!!!!!!!!!!!!!!!!!!!!!



% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));  %������weights
offset = nVars*nHidden(1);
for h = 2:length(nHidden)   %ȡ������Weights
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);  %�����Ȩ

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
    
    Backprop = [];  %�����Ĳв� !!!!!!!!!!!
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        gOutput = gOutput + repmat(fp{end}',[1,length(err)]) * diag(err);  % !!!!!!!!!!!!!!!!!!!

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            
            backprop = diag(err)*(outputWeights' * diag(sech(ip{end}).^2)); %!!!!!!!!!!!!!!!!!!
            gHidden{end} = gHidden{end} + repmat(fp{end-1}',[1,size(backprop,1)]) * backprop;  %!!!!!!!!!!!!!!!!!!!!!!

            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop; 
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
            
            Backprop = (backprop*inputWeights').*sech(X(i,:)).^2;  %�����в�  !!!!!!!!!!!!!!!!
            
        else  %������
           % Input Weights
           gInput = gInput + repmat(X(i,:)',[1,size(outputWeights',1)]) * (diag(err)*(outputWeights' * diag(sech(ip{end}).^2)));
           backprop = sum((diag(err)*(outputWeights' * diag(sech(ip{end}).^2))));  %!!!!!!!!!!!!!!
           Backprop = (backprop*inputWeights').*sech(X(i,:)).^2;  %�����в�!!!!!!!!!!!!!
           
        end
        
    end
    
    %kernal Ȩֵ����  !!!!!!!!!!!!!!!!!!!!
    A = reshape(Input(i,:),[sqrt(nv),sqrt(nv)]); % �ع�����ͼ��
    t = sqrt(length(Backprop));
    delta = reshape(Backprop,[t,t]);rot90(conv2(A,rot90(delta,2),'valid'),2);%�ع������в�ͼ��
    gKernal = rot90(conv2(A,rot90(delta,2),'valid'),2);%�ݶȼ���
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros((size(w,1)-lkernal^2),1);
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end

g = [reshape(gKernal,[lkernal^2,1]);g];
