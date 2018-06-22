% main.m

rng(2018);

dat = load('./MNIST.mat');

% preprocessing
trainX = reshape(dat.trainX, [size(dat.trainX, 1), 784]) / 255;
testX = reshape(dat.testX, [size(dat.testX, 1), 784]) / 255;
trainY = dat.trainY;
testY = dat.testY;

% for the training set
positives = find(vec2ind(trainY') == 5);       % number 4
negatives = find(vec2ind(trainY') == 10);      % number 9
pos = [ones(size(positives))', trainX(positives, :)];
neg = [-ones(size(negatives))', trainX(negatives, :)];
trainSet = datasample([pos; neg], 500, 1);

trainX = trainSet(:, 2:end);
trainY = trainSet(:, 1);

% for the testing set
positives = find(vec2ind(testY') == 5);       % number 4
negatives = find(vec2ind(testY') == 10);      % number 9
pos = [ones(size(positives))', testX(positives, :)];
neg = [-ones(size(negatives))', testX(negatives, :)];
testSet = [pos; neg];

testX = testSet(:, 2:end);
testY = testSet(:, 1);

M = 200;
[trainErr, testErr] = adaboost(trainX, trainY, testX, testY, 200);

h = figure('rend', 'painters', 'pos',[10, 10, 900, 600]);
plot(1:M, trainErr, 'r', 1:M, testErr, 'g');
xlabel('iterations');
ylabel('error in percentage');
legend('Training set', 'Testing set');
title('Error rates with 200 iterations');
saveas(h, 'error', 'png');

temp = [30; 100; 200];
disp([temp, trainErr(temp), testErr(temp)]);

save('result.mat', 'trainErr', 'testErr');
close all;

% use the Classification Learner Toolbox

[FineTree, ValidationErrorFineTree] = trainClassifierFineTree(trainSet);
yFineTree = FineTree.predictFcn(testX);
TestingErrorFineTree = sum(yFineTree ~= testY) / length(testY);
disp(sprintf('Fine Tree: validation error %f, testing error %f', ...
    1 - ValidationErrorFineTree, TestingErrorFineTree));

[MediumTree, ValidationErrorMediumTree] = trainClassifierMediumTree(trainSet);
yMediumTree = MediumTree.predictFcn(testX);
TestingErrorMediumTree = sum(yMediumTree ~= testY) / length(testY);
disp(sprintf('Medium Tree: validation error %f, testing error %f', ...
    1 - ValidationErrorMediumTree, TestingErrorMediumTree));

[CoarseTree, ValidationErrorCoarseTree] = trainClassifierCoarseTree(trainSet);
yCoarseTree = CoarseTree.predictFcn(testX);
TestingErrorCoarseTree = sum(yCoarseTree ~= testY) / length(testY);
disp(sprintf('Coarse Tree: validation error %f, testing error %f', ...
    1 - ValidationErrorCoarseTree, TestingErrorCoarseTree));

[BaggedTrees_30, ValidationErrorBaggedTrees_30] = trainClassifierBaggedTrees_30(trainSet);
yBaggedTrees_30 = BaggedTrees_30.predictFcn(testX);
TestingErrorBaggedTrees_30 = sum(yBaggedTrees_30 ~= testY) / length(testY);
disp(sprintf('Bagged Trees 30: validation error %f, testing error %f', ...
    1 - ValidationErrorBaggedTrees_30, TestingErrorBaggedTrees_30));

[BaggedTrees_100, ValidationErrorBaggedTrees_100] = trainClassifierBaggedTrees_100(trainSet);
yBaggedTrees_100 = BaggedTrees_100.predictFcn(testX);
TestingErrorBaggedTrees_100 = sum(yBaggedTrees_100 ~= testY) / length(testY);
disp(sprintf('Bagged Trees 100: validation error %f, testing error %f', ...
    1 - ValidationErrorBaggedTrees_100, TestingErrorBaggedTrees_100));

[BaggedTrees_300, ValidationErrorBaggedTrees_300] = trainClassifierBaggedTrees_300(trainSet);
yBaggedTrees_300 = BaggedTrees_300.predictFcn(testX);
TestingErrorBaggedTrees_300 = sum(yBaggedTrees_300 ~= testY) / length(testY);
disp(sprintf('Bagged Trees 300: validation error %f, testing error %f', ...
    1 - ValidationErrorBaggedTrees_300, TestingErrorBaggedTrees_300));

save('models.mat', ...
    'FineTree', 'MediumTree', 'CoarseTree', ...
    'BaggedTrees_30', 'BaggedTrees_100', 'BaggedTrees_300');