function p = Predict(theta, X)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);

h = Sigmoid(X * theta);
h(h >= 0.5) = 1;
h(h < 0.5) = 0;
p = h;

end
