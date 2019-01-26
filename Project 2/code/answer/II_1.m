clear all
load mnist_train
%load mnist_train_small
load mnist_valid
load mnist_test
Accuracy = zeros(1,5)
i = 0;
for k = [1, 3, 5, 7, 9]
    i = i+1;
    forc = run_knn(k, train_inputs, train_targets, valid_inputs);  %ʹ��knn������Ԥ��ֵ
    Accuracy(i) = sum(valid_targets == forc)/ size(valid_targets , 1);
end
plot([1, 3, 5, 7, 9],Accuracy,'-o');

k_hua = 5; %ѡȡk*Ϊ5
%�ڲ��Լ��ϲ鿴Ч��

Acc = zeros(1,3)
K = [k_hua-2, k_hua, k_hua+2 ];
K
for i = K
    forc = run_knn( i, train_inputs, train_targets, test_inputs);  %ʹ��knn������Ԥ��ֵ
    ite = (i-k_hua+2)/2+1;
    Acc(ite) = sum(test_targets == forc)/ size(test_targets , 1);
end
Acc
figure
plot(K,Acc,'-*');


