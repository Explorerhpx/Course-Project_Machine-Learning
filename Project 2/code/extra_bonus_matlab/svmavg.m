function [model] = svmavg(X,y,lambda,maxIter)

% Add bias variable
% ���bias
[n,d] = size(X);
X = [ones(n,1) X];

% Matlab indexes by columns,
%  so if we are accessing rows it will be faster to use  the traspose
Xt = X';

% Initial values of regression parameters
% ��ʼ��w
w = zeros(d+1,1);
tem = zeros(d+1,1);
% Apply stochastic gradient method
for t = 1:maxIter   %��������
    %if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)   %��ӡ�������̣��ص���������
        
     %   objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);  %����С����Ŀ�꺯����ֵ
      %  semilogy([0:t/n],objValues);      %��ͼ
      %  pause(.1);
    %end
    
    % ���ѡ��һ��training data
    % Pick a random training example
    i = ceil(rand*n);
    
    % ����sub - gradient
    % Compute sub-gradient
    [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i);
    
    % Set step size
    % ���ò���
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient step
    % �ݶ��½�
    w = w - alpha*(sg + lambda*w);
    tem = tem + w;
end

% ѵ�������� �ռ�����
model.w = tem / maxIter;
model.predict = @predict;

end

% ���ຯ��
function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

% ����sub_gradient�ĺ���
function [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i)

[d,n] = size(Xt);

% Function value
wtx = w'*Xt(:,i);
loss = max(0,1-y(i)*wtx);
f = loss;

% Subgradient
if loss > 0
    sg = -y(i)*Xt(:,i);
else
    sg = sparse(d,1);
end
end

