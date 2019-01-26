%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_valid;
load mnist_test;

%% TODO: Initialize hyperparameters.  %��ʼ����������������ѧϰ�ʣ�����ʼ
% Learning rate
hyperparameters.learning_rate = 0.05;
% Weight regularization parameter
hyperparameters.weight_regularization = 0;
% Number of iterations
hyperparameters.num_iterations = 1000;
% Logistics regression weights
% TODO: Set random weights.
weights = randn(size(train_inputs_small , 2) + 1, 1)/100;


%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% ��֤�󵼵���ȷ��
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

N = size(train_inputs_small,1);

%% Begin learning with gradient descent.
%�����ݶ��½�ѧϰ
%��ͼ��Ҫ������
train_ce = zeros(hyperparameters.num_iterations,1);
train_frac = zeros(hyperparameters.num_iterations,1);
neg_log_likehood = zeros(hyperparameters.num_iterations,1);
valid_ce = zeros(hyperparameters.num_iterations,1);
valid_frac = zeros(hyperparameters.num_iterations,1);

for t = 1:hyperparameters.num_iterations

	%% TODO: You will need to modify this loop to create plots etc.

	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = logistic(weights, ...
                                    train_inputs_small, ...
                                    train_targets_small, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);

	% Find the fraction of correctly classified validation examples.
	[temp, temp2, frac_correct_valid] = logistic(weights, ...
                                                 valid_inputs, ...
                                                 valid_targets, ...
                                                 hyperparameters);

    if isnan(f) || isinf(f)
		error('nan/inf error,try to reduce your learning rate or initialize with smaller weights');
	end

    %% Update parameters.
    weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
        
	%% Print some stats.
	fprintf(1, '��������:%4i   -log_likelihood:%4.2f TRAIN������%.6f TRAIN��ȷ��:%2.2f VALIC������ %.6f VALID��ȷ��:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
    %�ռ�����
    train_ce(t) = cross_entropy_train ;
    train_frac(t) = frac_correct_train*100;
    neg_log_likehood(t) = f/N;
    valid_ce(t) = cross_entropy_valid;
    valid_frac(t) = frac_correct_valid*100;

end

%��test������������
    predictions_test = logistic_predict(weights, test_inputs);
    [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
    fprintf(1, 'Test������%.6f Test������:%2.2f\n',cross_entropy_test, (1-frac_correct_test)*100);

%% ��ͼ
subplot(2,2,1)
plot(1:hyperparameters.num_iterations,neg_log_likehood);
title('negative log likelihood')
legend('Negative log likelihood');
subplot(2,2,2)
plot(1:hyperparameters.num_iterations,train_ce,1:hyperparameters.num_iterations,train_frac,1:0.1:hyperparameters.num_iterations,100,'r');
title('Data on training set')
legend('Cross entropy','Accuracy');
subplot(2,2,3)
plot(1:hyperparameters.num_iterations,valid_ce,1:hyperparameters.num_iterations,valid_frac,1:0.1:hyperparameters.num_iterations,100,'r');
title('Data on validation set')
legend('Cross entropy','Acuracy');
subplot(2,2,4)
plot(1:hyperparameters.num_iterations,valid_ce,1:hyperparameters.num_iterations,train_ce);
title('Cross entropy')
legend('Valid set','Training set');




