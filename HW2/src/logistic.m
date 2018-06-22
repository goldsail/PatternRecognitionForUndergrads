global X;
global Y;
global N;
global P;
global indeces_training;
global indeces_validation;

% read in data
dat = load('Breast_Cancer_Wisconsin_data.txt');
X = dat(:, 1:(end-1));
X = zscore(X);
Y = dat(:, end);

X = [ones(N, 1), X];
Y = (Y - 2) / 2;

N = size(X, 1);
P = size(X, 2);

% select training set and validation set

rng(2018);
indeces_training = randperm(N, round(0.7 * N));
indeces_validation = setdiff(1:N, indeces_training);

% logistic regression

theta0 = randn(P, 1);
theta = fminunc(@logistic_loss, theta0);

x = X(indeces_validation, :);
y = Y(indeces_validation, :);
yp = 1 ./ (1 + exp(- x * theta));
disp(sum((y - 0.5).*(yp - 0.5)<0) / size(x, 1)); % error rate
