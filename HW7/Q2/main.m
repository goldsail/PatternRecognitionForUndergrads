% main.m

addpath('tSNE_matlab');

rng(2018);

% preprocessing
trainX = load('trainX.txt') / 255;
testX = load('testX.txt') / 255;
trainY = load('trainY.txt');
testY = load('testY.txt');

X = trainX;
Y = trainY + 1;

% t-SNE
h = figure();
X_tsne = tsne(X, []);
scatter(X_tsne(:,1), X_tsne(:,2), Y, Y);
title('t-SNE');
saveas(h, 't-SNE', 'png');

% LLE
for k = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50]
    h = figure();
    X_lle = lle(X', k, 2)';
    scatter(X_lle(:,1), X_lle(:,2), Y, Y);
    title(sprintf('LLE with k = %d', k));
    saveas(h, sprintf('LLE_%d', k), 'png');
end

close all;

% train a SVM classifier

trainN = size(trainX, 1);
testN = size(testX, 1);

dat_raw = [Y, X];
X_full = tsne([trainX; testX], [], 2);
dat_red = [Y, X_full(1:trainN, :)];

test_raw = testX;
test_red = X_full(trainN + (1:testN), :);

classifierFull = trainClassifierFull(dat_raw);
classifierReduced = trainClassifierReduced(dat_red);

t_raw = classifierFull.predictFcn(test_raw);
t_red = classifierReduced.predictFcn(test_red);

disp(sprintf('test accuracy full: %f', sum(t_raw == testY + 1) / length(testY)));
disp(sprintf('test accuracy reduced: %f', sum(t_red == testY + 1) / length(testY)));


