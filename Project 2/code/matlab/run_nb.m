% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

load mnist_train_small;
load mnist_test;

% Add your code here (it should be less than 10 lines)
% ѵ��ģ��
[log_prior, class_mean, class_var] = train_nb(train_inputs_small, train_targets_small);
% ��ѵ�����ϵ�Ԥ��Ч��
[prediction_train, accuracy_train] = test_nb(train_inputs_small, train_targets_small, log_prior, class_mean, class_var);
% �ڲ��Լ��ϵ�Ԥ��Ч��
[prediction_test, accuracy_test] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var);
fprintf('ѵ��������ȷ��:%2.2f\n���Լ�����ȷ��:%2.2f\n',accuracy_train*100,accuracy_test*100);

Figure = zeros(56);
Figure(1:28,1:28) = reshape(class_mean( 1,:), 28, 28)';
Figure(1:28,29:56) = reshape(class_var( 1,:), 28, 28)';
Figure(29:56,1:28) = reshape(class_mean( 2,:), 28, 28)';
Figure(29:56,29:56) = reshape(class_var( 2,:), 28, 28)';
imagesc(Figure); 

