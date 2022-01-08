% Reza Ghorbani - January 2022
% Implementation of Regularized logistic regression

clc ;
clear ; 
close all ; 

data = load('data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

pos = find(y == 1);
neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'b+','LineWidth', 1); hold on;
plot(X(neg, 1), X(neg, 2), 'ro','LineWidth', 1);
xlabel('Feature 1');ylabel('Feature 2');
legend('Data 1', 'Data 2');
% Add Polynomial Features
X = Map_Feature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
% test_theta = ones(size(X,2),1);

lambda = 0.002; % Set regularization parameter lambda

[int_cost, int_grad] = Cost_Function_Reg(initial_theta, X, y, lambda);

% Optimizing using fminunc
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 200);

[theta, J, exit_flag] = fminunc(@(t)(Cost_Function_Reg(t, X, y, lambda)), initial_theta, options);

% Plot Decision Boundary
figure;
pos = find(y == 1);
neg = find(y == 0);
X_2 = X(:,2:3);
plot(X_2(pos, 1), X_2(pos, 2), 'b+','LineWidth', 1); hold on;
plot(X_2(neg, 1), X_2(neg, 2), 'ro','LineWidth', 1); 
u = linspace(-1, 1.5, 50);
v = linspace(-0.8, 1.2, 50);
z = zeros(length(u), length(v));

% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = Map_Feature(u(i), v(j))*theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% You need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 1)
xlabel('Feature 1');ylabel('Feature 2');
legend('Data 1', 'Data 2','Decision Boundary');

% Compute accuracy on our training set
p = Predict(theta, X);
Accuracy = mean(double(p == y)) * 100;
