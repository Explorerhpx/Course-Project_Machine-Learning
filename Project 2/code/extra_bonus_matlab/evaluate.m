function [ce, frac_correct] = evaluate(targets, y)   %输出targets和y的交叉熵/正确分类的比值
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function
ce = sum(-log(y).*targets);
correct = sum((y >= 0.5) == targets);
frac_correct = correct/size(y ,1);
end
