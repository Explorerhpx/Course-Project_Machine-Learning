function [model] = svmavg(X,y,lambda,maxIter)

% Add bias variable
% 添加bias
[n,d] = size(X);
X = [ones(n,1) X];

% Matlab indexes by columns,
%  so if we are accessing rows it will be faster to use  the traspose
Xt = X';

% Initial values of regression parameters
% 初始化w
w = zeros(d+1,1);
tem = zeros(d+1,1);
% Apply stochastic gradient method
for t = 1:maxIter   %迭代次数
    %if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)   %打印收敛过程，关掉可以提速
        
     %   objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);  %（最小化）目标函数的值
      %  semilogy([0:t/n],objValues);      %绘图
      %  pause(.1);
    %end
    
    % 随机选择一个training data
    % Pick a random training example
    i = ceil(rand*n);
    
    % 计算sub - gradient
    % Compute sub-gradient
    [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i);
    
    % Set step size
    % 设置步长
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient step
    % 梯度下降
    w = w - alpha*(sg + lambda*w);
    tem = tem + w;
end

% 训练结束， 收集数据
model.w = tem / maxIter;
model.predict = @predict;

end

% 分类函数
function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

% 计算sub_gradient的函数
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

