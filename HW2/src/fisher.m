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

Y = (Y - 2) / 2;

N = size(X, 1);
P = size(X, 2);

% select training set and validation set

rng(2018);
indeces_training = randperm(N, round(0.7 * N));
indeces_validation = setdiff(1:N, indeces_training);

% Fisher discriminant

% on training set

x = X(indeces_training, :);
y = Y(indeces_training, :);

N1 = sum(double(y > 0.5)); % positive samples
N2 = sum(double(y < 0.5)); % negative samples

X1 = x(y > 0.5, :);
Y1 = y(y > 0.5, :);
X2 = x(y < 0.5, :);
Y2 = x(y < 0.5, :);

m1 = mean(X1, 1)';
m2 = mean(X2, 1)';

Sw = (X1' - repmat(m1, 1, N1)) * (X1' - repmat(m1, 1, N1))' ...
    + (X2' - repmat(m2, 1, N2)) * (X2' - repmat(m2, 1, N2))';
Sb = (m1 - m2) * (m1 - m2)';

w = (Sw + N1 * N2 * Sb / N) \ (N * (m1 - m2));
w0 = - w' * mean(x, 1)';

% on validation set
x = X(indeces_validation, :);
y = Y(indeces_validation, :);

yp = x * w + w0;
disp(sum(yp .* y < 0) / size(x, 1)); % error rate

