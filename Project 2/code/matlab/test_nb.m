function [prediction, accuracy] = test_nb(test_data, test_label, log_prior, class_mean, class_var)
% Test a learned Naive Bayes classifier.
%
% Usage:
%   [prediction, accuracy] = test_nb(test_data, test_label, log_prior, class_mean, class_var)
%
% test_data:  N_examples x M_dimensions matrix
% test_label: N_examples x 1 binary label vector
% log_prior: 2 x 1 vector, log_prior(i) = log p(C=i)
% class_mean: 2 x M_dimensions  matrix, class_mean(:,i) is the mean vector for class i.
% class_var: 2 x M_dimensions matrix, class_var(:,i) is the variance vector for class i.
%
% prediction: N_examples x 1 binary label vector
% accuracy: a real number
%

K = length(log_prior);  %类别数
n_examples = size(test_data, 1);

log_prob = zeros(n_examples, K);  %储存每一个example属于每一个class中的概率

for k = 1 : K
    mean_mat = repmat(class_mean(k, :), [n_examples, 1]);
    var_mat = repmat(class_var(k, :), [n_examples, 1]);
    log_prob(:, k) = sum(-0.5 * (test_data - mean_mat).^2 ./ var_mat - 0.5 * log(var_mat), 2) + log_prior(k);
end

[~, prediction] = max(log_prob, [], 2);
prediction = prediction - 1;
accuracy = mean(prediction == test_label);

return
end

