clear all

load quantum.mat
[n,d] = size(X);

% Split into training and validation set
perm = randperm(n);
Xvalid = X(n/2+1:end,:);
yvalid = y(n/2+1:end);
X = X(1:n/2,:);
y = y(1:n/2);
n = n/2;

Accu = zeros(1,3) ;
ite = 0 ;
for i = [50,300,500,2000]
    ite = ite + 1
    lambda = 1/n;
    model = svm_2(X,y,lambda,i*n);
    % ½øÐÐÔ¤²â
    pre = model.predict(model,Xvalid);
    % ÆÀ¼Û
    [ce, accu] = evaluate(yvalid, pre);
    Accu(ite) = accu;
end

plot([1:4],Accu)
title('Accuracy')
xlabel('n');
set(gca,'Xticklabel',{'50','','300','','500','','2000'});


