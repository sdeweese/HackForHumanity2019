net = alexnet % load alexnet
 
imds = imageDatastore('/Images/Training', 'IncludeSubfolders',true,"LabelSource","Foldernames"); % create data store for training
imds2 = imageDatastore('/Images/Predictions/', 'IncludeSubfolders',true,"LabelSource","Foldernames"); % create data store for the predictinos

names = imds.Labels
numClasses = numel(categories(imds.Labels));

inpSz = [227 227];

trainImgs = augmentedImageDatastore(inpSz,imds,'ColorPreprocessing','gray2rgb')
testImgs = augmentedImageDatastore(inpSz,imds2,'ColorPreprocessing','gray2rgb')
  
% start modifying pretrained network using transfer learning
layers = net.Layers
% create new layer with types of outcomes i.e. cancer or notcancer
fc = fullyConnectedLayer(numClasses)
% replace the 23rd layer with our new layer of new training data
layers(end-2) = fc
% replace the last layer with our new classifcation
layers(end) = classificationLayer
 
% set training options
options = trainingOptions('sgdm','InitialLearnRate',0.001)
 
[net,info] = trainNetwork(trainImgs, layers, options);
 
testpreds = classify(net,testImgs);
