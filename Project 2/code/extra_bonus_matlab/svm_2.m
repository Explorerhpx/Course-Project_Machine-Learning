function [model] = svm_2(X,y,lambda,maxIter)

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
helf = fix(maxIter/2);
objValues = 100; %������ý����objValues
w_tem = zeros(d+1,1);
% Apply stochastic gradient method
for t = 1:maxIter   %��������
    % ���ѡ��һ��training data
    % Pick a random training example
    i = ceil(rand*n);
    
    % ����sub - gradient
    % Compute sub-gradient
    [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i);
    
    % Set step size
    % ���ò���
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient ste
    w = w - alpha*(sg + lambda*w);
    %������
    if mod(t-1,(n/10)) == 0
        objtem = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);  %����С����Ŀ�꺯����ֵ
        if objtem < objValues
            objValues = objtem;
            w_tem = w;
        end
    end
    %����ֵ��
    if t >= helf
        tem = tem + w;
    end
end

% �ж�ͨ�����ַ����ռ��������Ǹ����
tem = tem/(helf+1);
% ���ĵ������
obj1 = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);
% ʹ��ƽ��ֵ���õ��Ľ��
obj2 = (1/n)*sum(max(0,1-y.*(X*tem))) + (lambda/2)*(tem'*tem);
% ʹ�ó������õ��Ľ��
obj3 = (1/n)*sum(max(0,1-y.*(X*w_tem))) + (lambda/2)*(w_tem'*w_tem);
least = min([obj1,obj2,obj3]);
if least == obj1
    model.w = w;
else
    if least == obj2
        model.w = tem;
    else
        model.w = w_tem;
    end
end
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

