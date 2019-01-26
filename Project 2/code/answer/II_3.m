%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_valid;
load mnist_test;

%% Global variables
%L = [0.001, 0.01, 0.1, 1.0];  %candidate of lam
L = [1.0];  %candidate of lam
Train_Cross = zeros(size(L,2),1);
Train_error = zeros(size(L,2),1);
Val_Cross = zeros(size(L,2),1);
Val_error = zeros(size(L,2),1);
rept_times = 20; %每个lambda需要重复regression的次数

%% 执行过程
for ite = 1 : size(L,2)
    lam = L(ite);
    fprintf('lambda: %1.3f\n',lam);
    train_ce = zeros(rept_times,1);
    train_frac = zeros(rept_times,1);
    valid_ce = zeros(rept_times,1);
    valid_frac = zeros(rept_times,1);
    
    %% 进行regression
    for time = 1:rept_times  %每个lam重复rept_times次
        
        fprintf('第%d个lambda   第%d次拟合\n',ite,time);
        
        %% TODO: Initialize hyperparameters.  %初始化超参数：步长（学习率）、初始
        hyperparameters.learning_rate = 0.05;
        hyperparameters.weight_regularization = lam;
        hyperparameters.num_iterations = 2500;
        weights = randn(size(train_inputs_small , 2) + 1, 1)/100;
        
        %% 验证求导的正确性
        nexamples = 20;
        ndimensions = 10;
        diff = checkgrad('logistic_pen', ...
            randn((ndimensions + 1), 1), ...   % weights
            0.001,...                          % perturbation
            randn(nexamples, ndimensions), ... % data
            rand(nexamples, 1), ...            % targets
            hyperparameters);                   % other hyperparameters
        
        N = size(train_inputs_small,1);
        
        %% 进行梯度下降学习
        for t = 1:hyperparameters.num_iterations
            
            % Find the negative log likelihood and derivative w.r.t. weights.
            [f, df, predictions] = logistic_pen(weights, ...
                train_inputs_small, ...
                train_targets_small, ...
                hyperparameters);
            
            [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);
            
            % Find the fraction of correctly classified validation examples.
            [temp, temp2, frac_correct_valid] = logistic_pen(weights, ...
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
            %fprintf(1, '迭代次数:%4i   -log_likelihood:%4.2f TRAIN交叉熵%.6f TRAIN正确率:%2.2f VALIC交叉熵 %.6f VALID正确率:%2.2f\n',...
            %t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
            
            %% 收集数据
            if t == hyperparameters.num_iterations  %学习过程完毕
                train_ce(time) = cross_entropy_train;
                train_frac(time) = frac_correct_train *100;
                valid_ce(time) = cross_entropy_valid;
                valid_frac(time) = frac_correct_valid*100;
            end
        end
    end
    
    %收集这个lam的所有数据
    Train_Cross(ite) = mean(train_ce);
    Train_error(ite) = 100 - mean(train_frac);
    Val_Cross(ite) = mean(valid_ce);
    Val_error(ite) = 100 - mean(valid_frac);
    
end

predictions_test = logistic_predict(weights, test_inputs);
[cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
fprintf(1, 'Test交叉熵%.6f Test错误率:%2.2f\n',cross_entropy_test, (1-frac_correct_test)*100);

Train_Cross'
Train_error'
Val_Cross'
Val_error'
subplot(1,2,1);
plot(1:4,Train_Cross,'o-',1:4,Val_Cross,'*-');
title('Cross Entropy');
legend('Train','Val');
xlabel('\lambda');
set(gca,'Xticklabel',{'0.001','0.01','0.1','1.0'});
ylim( [-1 , 10]);
subplot(1,2,2);
plot(1:4,Train_error,'o-',1:4,Val_error,'*-');
title('Classfication Error');
legend('Train','Val');
xlabel('\lambda');
set(gca,'Xticklabel',{'0.001','0.01','0.1','1.0'});
ylim( [-5 , 50]);

