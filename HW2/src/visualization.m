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

N1 = Y > 0.5;
N2 = Y < 0.5;

[coeff, score, latent] = pca(X);

Xp = score(:, 1:2);
figure();
scatter(Xp(:, 1), Xp(:, 2), 5, N1');
axis([-4, 8, -3, 2]);
title('First two principal components');
