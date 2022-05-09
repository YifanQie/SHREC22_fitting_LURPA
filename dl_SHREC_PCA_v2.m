% clear;
% clc;

% all the figures in the dataset
digitDatasetPath = ['D:\matlab workplace\partition\Measurement test\DL_data_trainingset']; % surface data


% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% count how may figures we have
labelCount = countEachLabel(imds)
figurenumber = sum(labelCount.Count)

% % -----------------------------------------------------
% % We will use 'read' function to read images from the datastore
% % First call of read returns first image
% % Second call of read returns second image, and so on

% % -----------------------------------------------------
% % Now from this point, use the custom reader function
imds.ReadFcn = @customreader;

% % Reset the datastore to the state where no data has been read from it.
reset(imds);

figure;
perm = randperm(figurenumber,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{i});
end

% % size of one figure
% img = readimage(imds,1);
% size(img)

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.95,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:20
    subplot(4,5,i);
    imshow(imdsTrain.Files{i});
end

net = alexnet;
analyzeNetwork(net);

inputSize = net.Layers(1).InputSize

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% % -----------------------------------------------------
% % Code of custom image datastore reader function
function data = customreader(filename)
    onState = warning('off', 'backtrace');
    c = onCleanup(@() warning(onState));
    data = imread(filename);
    data = data(:,:,min(1:3, end)); 
    data = imresize(data, [227 227]);
end
% % -----------------------------------------------------