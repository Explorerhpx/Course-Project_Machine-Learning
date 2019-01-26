function [model] = svm_2(X,y,lambda,maxIter)

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
helf = fix(maxIter/2);
objValues = 100; %储存最好结果的objValues
w_tem = zeros(d+1,1);
% Apply stochastic gradient method
for t = 1:maxIter   %迭代次数
    % 随机选择一个training data
    % Pick a random training example
    i = ceil(rand*n);
    
    % 计算sub - gradient
    % Compute sub-gradient
    [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i);
    
    % Set step size
    % 设置步长
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient ste
    w = w - alpha*(sg + lambda*w);
    %抽样法
    if mod(t-1,(n/10)) == 0
        objtem = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);  %（最小化）目标函数的值
        if objtem < objValues
            objValues = objtem;
            w_tem = w;
        end
    end
    %后半均值法
    if t >= helf
        tem = tem + w;
    end
end

% 判断通过三种方法收集的数据那个最好
tem = tem/(helf+1);
% 最后的迭代结果
obj1 = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);
% 使用平均值法得到的结果
obj2 = (1/n)*sum(max(0,1-y.*(X*tem))) + (lambda/2)*(tem'*tem);
% 使用抽样法得到的结果
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

