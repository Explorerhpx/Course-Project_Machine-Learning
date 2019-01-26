clear all
load digits.mat
[n,d] = size(X); 

Nsample = 400; %共添加1200个样本
X_self = X;
y_self = y;

Tem = randperm(n);
tem = Tem(1:Nsample);
sample_x = X(tem,:);%随机抽取400个样本
sample_y = y(tem);

y_self = [y_self;sample_y];  %添加训练标签

%旋转
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    dushu = -10 + 20*rand(1);  %旋转度数为-8 到 8度之间的随机数
    B = imrotate(A,dushu,'crop');
    X_self = [X_self;reshape(B,[1,256])]; %添加训练样本
end

tem = Tem(Nsample+1:2 * Nsample);
sample_x = X(tem,:);
sample_y = y(tem);

y_self = [y_self;sample_y];  %添加训练标签

%缩放
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    rate = 0.85 + 0.3 * rand(1);   %图形缩放比例在0.8,1.2之间
    b = imresize(A,rate,'bilinear');
    B = zeros(16,16);
    if rate<1   %使缩放后的图像和原来大小相同
        k = size(b,1);
        fill = ceil((16-k)/2);
        B(fill+1:(fill+k),fill+1:(fill+k)) = b;
    elseif rate >1
        qu = ceil((rate-1)*8);
        B = b(qu:(qu+15),qu:(qu+15));
    end
    X_self = [X_self;reshape(B,[1,256])]; %添加训练样本
end

tem = Tem(2*Nsample+1:3 * Nsample);
sample_x = X(tem,:);
sample_y = y(tem);

y_self = [y_self;sample_y];  %添加训练标签

%平移
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    heng = round(-2 + 4 * rand(1));   %图形左右平移两格以内
    shu = round(-2 + 4 * rand(1)); %图形上下平移两格以内
    B = zeros(16,16);
    if heng <= 0 %左移
        if shu <= 0 %下移
            B((1-shu):16,1:(16+heng)) = A(1:(16+shu),(1-heng):16);
        else %上移
            B(1:(16-shu),1:(16+heng)) = A((1+shu):16,(1-heng:16));
        end
    else %右移
        if shu <= 0 %下移
            B((1-shu):16,(1+heng):16) = A(1:(16+shu),1:(16-heng));
        else %上移
            B(1:(16-shu),(1+heng):16) = A((1+shu):16,1:(16-heng));
        end
    end
    X_self = [X_self;reshape(B,[1,256])]; %添加训练样本
end
        
save mydata X_self y_self  %存储新样本
