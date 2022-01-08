function [J, grad] = Cost_Function_Reg(theta, X, y, lambda)

m = length(y); % number of training examples
 
J = 0;
grad = zeros(size(theta));

A = X';
h = Sigmoid(X * theta);
J = ((-1/m) * ( (y' * log(h)) + (1-y') * (log(1 - h)))) + ((lambda/(2 * m)) * sum(theta(2:size(theta,1),1) .^ 2));
grad = ((1 / m) * (X' * (h - y))) + ((lambda / m) * theta);
grad(1, 1) = ((1 / m) * (A(1, :) * (h - y)));

end
