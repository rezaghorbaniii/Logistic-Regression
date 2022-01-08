% Reza Ghorbani - January 2022
% Implementation of Logistic Regression

clc ;
clear ; 
close all; 

data = load('data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

pos = find(y == 1);
neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'b+','LineWidth', 1); hold on;
plot(X(neg, 1), X(neg, 2), 'ro','LineWidth', 1);
xlabel('Feature 1');ylabel('Feature 2');
legend('Data 1', 'Data 2');

[m, n] = size(X);

X = [ones(m, 1) X]; % Add intercept term to x and X_test

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
% test_theta = [-24; 0.2; 0.2];
[int_cost, int_gradient] = Cost_Function(initial_theta, X, y);

% Optimizing using fminunc
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

[theta, cost] = fminunc(@(t)(Cost_Function(t, X, y)), initial_theta, options);

% Plot Decision Boundary
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

figure;
pos = find(y == 1);
neg = find(y == 0);
X_2 = X(:,2:3);
plot(X_2(pos, 1), X_2(pos, 2), 'b+','LineWidth', 1); hold on;
plot(X_2(neg, 1), X_2(neg, 2), 'ro','LineWidth', 1); 
plot(plot_x, plot_y,'LineWidth', 1);   
axis([30, 100, 30, 100]);
xlabel('Feature 1');ylabel('Feature 2');
legend('Data 1', 'Data 2','Decision Boundary');

% Predict data probability
prob = Sigmoid([1 45 85] * theta);

% Compute accuracy on training set
p = Predict(theta, X);
Accuracy = mean(double(p == y)) * 100;
