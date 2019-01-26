clear all
load digits.mat
[n,d] = size(X); 

Nsample = 400; %�����1200������
X_self = X;
y_self = y;

Tem = randperm(n);
tem = Tem(1:Nsample);
sample_x = X(tem,:);%�����ȡ400������
sample_y = y(tem);

y_self = [y_self;sample_y];  %���ѵ����ǩ

%��ת
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    dushu = -10 + 20*rand(1);  %��ת����Ϊ-8 �� 8��֮��������
    B = imrotate(A,dushu,'crop');
    X_self = [X_self;reshape(B,[1,256])]; %���ѵ������
end

tem = Tem(Nsample+1:2 * Nsample);
sample_x = X(tem,:);
sample_y = y(tem);

y_self = [y_self;sample_y];  %���ѵ����ǩ

%����
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    rate = 0.85 + 0.3 * rand(1);   %ͼ�����ű�����0.8,1.2֮��
    b = imresize(A,rate,'bilinear');
    B = zeros(16,16);
    if rate<1   %ʹ���ź��ͼ���ԭ����С��ͬ
        k = size(b,1);
        fill = ceil((16-k)/2);
        B(fill+1:(fill+k),fill+1:(fill+k)) = b;
    elseif rate >1
        qu = ceil((rate-1)*8);
        B = b(qu:(qu+15),qu:(qu+15));
    end
    X_self = [X_self;reshape(B,[1,256])]; %���ѵ������
end

tem = Tem(2*Nsample+1:3 * Nsample);
sample_x = X(tem,:);
sample_y = y(tem);

y_self = [y_self;sample_y];  %���ѵ����ǩ

%ƽ��
for i = 1:Nsample
    A = reshape(sample_x(i,:),[16,16]);
    heng = round(-2 + 4 * rand(1));   %ͼ������ƽ����������
    shu = round(-2 + 4 * rand(1)); %ͼ������ƽ����������
    B = zeros(16,16);
    if heng <= 0 %����
        if shu <= 0 %����
            B((1-shu):16,1:(16+heng)) = A(1:(16+shu),(1-heng):16);
        else %����
            B(1:(16-shu),1:(16+heng)) = A((1+shu):16,(1-heng:16));
        end
    else %����
        if shu <= 0 %����
            B((1-shu):16,(1+heng):16) = A(1:(16+shu),1:(16-heng));
        else %����
            B(1:(16-shu),(1+heng):16) = A((1+shu):16,1:(16-heng));
        end
    end
    X_self = [X_self;reshape(B,[1,256])]; %���ѵ������
end
        
save mydata X_self y_self  %�洢������
