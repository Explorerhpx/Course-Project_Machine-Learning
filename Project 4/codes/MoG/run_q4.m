clear all
[train, valid, test, target_train, target_valid, target_test] = load_data();
target_train = target_train ;
target_valid = target_valid ;
target_test = target_test;

load digits

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];

for k = 1 : 4
    K = numComponent(k);
    for ite = 1:20
        % Train a MoG model with K components for digit 2
        %-------------------- Add your code here --------------------------------
        [P2,mu2,vary2,logProbX] = mogEM_II(train2,K,30,1e-7,1e-6,0);
        
        % Train a MoG model with K components for digit 3
        %-------------------- Add your code here --------------------------------
        [P3,mu3,vary3,logProbX] = mogEM_II(train3,K,30,1e-7,1e-4,0);
        %-------------------- Add your code here --------------------------------
        
        % Caculate the probability P(d=1|x) and P(d=2|x),
        % classify examples, and compute the error rate
        % Hints: you may want to use mogLogProb function
        %-------------------- Add your code here --------------------------------
         % Do classification on train
        for i = 1:size(train,2)
            logProb2 = mogLogProb(P2,mu2,vary2,train(:,i));
            logProb3 = mogLogProb(P3,mu3,vary3,train(:,i));
            predict(i) = logProb3 > logProb2;  % do classification
        end
        errorTrain(k) = errorTrain(k) + sum(predict~= target_train)/size(train,2);
        
        % Do classification on validation
        predict = [];
        for i = 1:size(valid,2)
            logProb2 = mogLogProb(P2,mu2,vary2,valid(:,i));
            logProb3 = mogLogProb(P3,mu3,vary3,valid(:,i));
            predict(i) = logProb3 > logProb2;  % do classification
        end
        errorValidation(k) = errorValidation(k) + sum(predict~= target_valid)/size(valid,2);
        
        predict = [];
        % Do classification on test
        for i = 1:size(test,2)
            logProb2 = mogLogProb(P2,mu2,vary2,test(:,i));
            logProb3 = mogLogProb(P3,mu3,vary3,test(:,i));
            predict(i) = logProb3 > logProb2;  % do classification
        end
        errorTest(k) = errorTest(k) + sum(predict~= target_test)/size(test,2);
        
    end
end

% Plot the error rate
%-------------------- Add your code here --------------------------------

errorTrain = errorTrain/20;
errorValidation = errorValidation/20;
errorTest = errorTest/20;

plot(1:4,errorTrain,'.-',1:4,errorValidation,'.-',1:4,errorTest,'.-');
legend('Train','Validation','Test')
title('Error rate vs Number of mixture components')
