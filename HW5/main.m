% main.m

rng(2018);

dat = load('./MNIST.mat');

h = figure();
for i = 1:10
    j = datasample(find(dat.trainY(:, i) == 1), 1);
    subplot(1, 10, i);
    imshow(uint8(squeeze(dat.trainX(j, :, :))));
end
saveas(h, 'glance', 'png');

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
trainSet = [pos; neg];

% for the testing set
positives = find(vec2ind(testY') == 5);       % number 4
negatives = find(vec2ind(testY') == 10);      % number 9
pos = [ones(size(positives))', testX(positives, :)];
neg = [-ones(size(negatives))', testX(negatives, :)];
testSet = [pos; neg];
t = testSet(:, 1);

clear trainX trainY testX testY positives negatives pos neg

% Linear SVM
fprintf('\ntraining Linear SVM... ');
[trainedClassifierLinear, validationAccuracyLinear] = trainClassifierLinear(trainSet);
yLinear = trainedClassifierLinear.predictFcn(testSet(:, 2:end));
testAccuracyLinear = sum(yLinear == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyLinear);

% Quadratic SVM
fprintf('\ntraining Quadratic SVM... ');
[trainedClassifierQuadratic, validationAccuracyQuadratic] = trainClassifierQuadratic(trainSet);
yQuadratic = trainedClassifierQuadratic.predictFcn(testSet(:, 2:end));
testAccuracyQuadratic = sum(yQuadratic == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyQuadratic);

% Cubic SVM
fprintf('\ntraining Cubic SVM... ');
[trainedClassifierCubic, validationAccuracyCubic] = trainClassifierCubic(trainSet);
yCubic = trainedClassifierCubic.predictFcn(testSet(:, 2:end));
testAccuracyCubic = sum(yCubic == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyCubic);

% GaussianFine SVM
fprintf('\ntraining Gaussian Fine SVM... ');
[trainedClassifierGaussianFine, validationAccuracyGaussianFine] = trainClassifierGaussianFine(trainSet);
yGaussianFine = trainedClassifierGaussianFine.predictFcn(testSet(:, 2:end));
testAccuracyGaussianFine = sum(yGaussianFine == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyGaussianFine);

% GaussianMedium SVM
fprintf('\ntraining Gaussian Medium SVM... ');
[trainedClassifierGaussianMedium, validationAccuracyGaussianMedium] = trainClassifierGaussianMedium(trainSet);
yGaussianMedium = trainedClassifierGaussianMedium.predictFcn(testSet(:, 2:end));
testAccuracyGaussianMedium = sum(yGaussianMedium == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyGaussianMedium);

% GaussianCoarse SVM
fprintf('\ntraining Gaussian Coarse SVM... ');
[trainedClassifierGaussianCoarse, validationAccuracyGaussianCoarse] = trainClassifierGaussianCoarse(trainSet);
yGaussianCoarse = trainedClassifierGaussianCoarse.predictFcn(testSet(:, 2:end));
testAccuracyGaussianCoarse = sum(yGaussianCoarse == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyGaussianCoarse);

% Fully connected neural network (FCNN) with 100 hidden nodes
fprintf('\ntraining FCNN with 100 hidden nodes... ');
trainedClassifierFullyConnected = trainClassifierFullyConnected(trainSet);
yFullyConnected= vec2ind(trainedClassifierFullyConnected(testSet(:, 2:end)'))' * 2 - 3;
testAccuracyFullyConnected = sum(yFullyConnected == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyFullyConnected);

% Losgistic regression
fprintf('\ntraining logistic regression... ');
[trainedClassifierLogistic, validationAccuracyLogistic] = trainClassifierLogitic(trainSet);
yLogistic = trainedClassifierLogistic.predictFcn(testSet(:, 2:end));
testAccuracyLogistic = sum(yLogistic == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyLogistic);

% Naive Bayes
trainedClassifierBayes = fitcnb(trainSet(:, 2:end), trainSet(:, 1));
yBayes = predict(trainedClassifierBayes, testSet(:, 2:end));
testAccuracyBayes = sum(yBayes == t) / length(t);
fprintf('test accuracy: %f\n', testAccuracyBayes);

close all;