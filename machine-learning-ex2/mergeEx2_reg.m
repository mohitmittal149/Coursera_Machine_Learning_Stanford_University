%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
 
%% Initialization
clear ; close all; clc
 
%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
 
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
 
plotData(X, y);

1***********************************************************************

function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
 
% Create New Figure
figure; hold on;
 
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
 
% =========================================================================
hold off;
 
end

2***********************************************************

 
% Put some labels
hold on;
 
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
 
% Specified in plot order
legend('y = 1', 'y = 0')
hold off;
 
 
%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%
 
% Add Polynomial Features
 
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

1***********************************************************************

function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
 
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end
 
end
2***********************************************************************

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
 
% Set regularization parameter lambda to 1
lambda = 1;
 
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

1***********************************************************************

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
 
% Initialize some useful values
m = length(y); % number of training examples
 
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%            derivatives of the cost w.r.t. each parameter in theta
theta0 = [0; theta(2:end)];
z=X*theta;
h=sigmoid(z);
b=(-y)'*log(h)-(1-y)'*log(1-h);
J=((1/m).*sum(b))+(lambda/(2*m)).*sum(theta0.^2);
 
grad = (1/m) * (X' * (h - y) + lambda * theta0);
 
% =============================================================
 
end

2***********************************************************************
 
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');
 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
 
% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);
 
fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');
 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
 
%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%
 
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
 
% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;
 
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
 
% Optimize
[theta, J, exit_flag] = ...
    fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
 
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
 
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
 
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
 
% Compute accuracy on our training set
p = predict(theta, X);
 
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
 


