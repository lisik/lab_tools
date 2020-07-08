% code to categorize stimuli based on image features 
% im_features can be set to pixel intensity (RGB or grayscale) or output of
% different VGG layers (set layers below)
clear all
im_dir =  '~/Downloads/stimuli-no-scrubs/';
im_dir = '~/Downloads/experiment_recrop/experiment_recrop/';
% 2 classes of images, in separate directories
im_format = '.jpg';
resample_runs = 20;
class1_dir = [im_dir '/interact/'];
class2_dir = [im_dir '/non-interact/'];
class3_dir = [im_dir '/watch/'];
svm = 0;% svm or correlation coefficient classifier

im_features = 'vgg'; %'vgg'; %'gray'; %options: grayscale, rgb, vgg, resnet 
vgg_layers = {6,11,18,25,32};% 5 pooling layers in VGG are 06, 11, 18, 25, 32
resnet_layers = {'activation_1_relu' ,'activation_10_relu' ,'activation_22_relu','activation_40_relu' ,'activation_49_relu' };
run_pca = 1;
num_PCs = 50;

% number of SVM resample runs


%% Have to order these properly for MEG experiment...
class1 = dir([class1_dir '*' im_format]);
class1 = {class1.name};
class2 = dir([class2_dir '*' im_format]);
class2 = {class2.name};
class3 = dir([class3_dir '*' im_format]);
class3 = {class3.name};

if strcmp(im_features, 'vgg')
    % can modify here with AlexNet if preferred
    net = vgg16;
    imageSize = net.Layers(1).InputSize;
    layers = vgg_layers;
elseif strcmp(im_features, 'resnet')
    net = resnet50;
    imageSize = net.Layers(1).InputSize;
    layers = resnet_layers;
else
    layers = 1; %dummy variable for pixel case
end
acc= zeros(length(layers),resample_runs);
D = zeros(length(layers), length(class1)*2, length(class1)*2);

for l = 1:length(layers)
    clear class1_ims class2_ims class3_ims
for i = 1:length(class1)
    
    if strcmpi(im_features, 'gray')
    tmp1 = rgb2gray(imread([class1_dir class1{i}]));
    tmp2 = rgb2gray(imread([class2_dir class2{i}]));
    
    elseif strcmpi(im_features, 'rgb')
    tmp1 = imread([class1_dir class1{i}]);
    tmp2 = imread([class2_dir class2{i}]);
    
    elseif strcmpi(im_features, 'VGG') | strcmpi(im_features, 'resnet')
    layer = layers{l};
  
    tmp1 = imread([class1_dir class1{i}]);
    tmp1 = augmentedImageDatastore(imageSize, tmp1, 'ColorPreprocessing', 'gray2rgb');
    tmp1 = activations(net, tmp1, layer, 'OutputAs', 'columns');
    
    tmp2 = imread([class2_dir class2{i}]);
    tmp2 = augmentedImageDatastore(imageSize, tmp2, 'ColorPreprocessing', 'gray2rgb');
    tmp2 = activations(net, tmp2, layer, 'OutputAs', 'columns');
    
    if i<=length(class3)
    tmp3 = imread([class3_dir class3{i}]);
    tmp3 = augmentedImageDatastore(imageSize, tmp3, 'ColorPreprocessing', 'gray2rgb');
    tmp3 = activations(net, tmp3, layer, 'OutputAs', 'columns');
    class3_ims(i,:) = tmp3(:);
    end
    
    else
    error('Not a valid image feature \n options are rgb, gray, VGG, resnet')
    end
    
    class1_ims(i,:) = tmp1(:);
    class2_ims(i,:) = tmp2(:);
    
end

imageFeatures = double([class1_ims; class2_ims]);
labels = [ones(1,length(class1)), 2*ones(1,length(class2))];

if run_pca
imageFeatures_orig = [imageFeatures; class3_ims];
[coeff,imageFeatures_pca,latent,tsquared,explained,mu] = pca(imageFeatures_orig); 
imageFeatures = imageFeatures_pca(1:length(class1)*2,1:num_PCs);
end

%% z score data
imageFeatures = zscore(imageFeatures);

for i = 1:resample_runs
    % 80-20 crossvalidation split
    [train,test] = crossvalind('holdout',labels,0.2);
%     test = [(i-1)*4+1:i*4, (i-1)*4+25:i*4+24];
%    test = [(i-1)*4+1:i*4, (i-1)*4+17:i*4+16];

    inds = randperm(12);
    test= [(inds(1)-1)*2+1:inds(1)*2, (inds(2)-1)*2+1:inds(2)*2, ...
        (inds(1)-1)*2+25:inds(1)*2+24, (inds(2)-1)*2+25:inds(2)*2+24];
    train = setdiff(1:size(imageFeatures,1),test);
    train_labels = labels(train);
    test_labels = labels(test)';
    train_data = imageFeatures(train,:);
    test_data = imageFeatures(test,:);
    
    if svm
    SVMStruct = fitcsvm(train_data,train_labels);
    pred = predict(SVMStruct, test_data);
    acc(l,i) = sum(pred==test_labels)/length(test_labels);
    else
    clear mean_train
    mean_train(1,:) = mean(train_data(train_labels==1,:));
    mean_train(2,:) = mean(train_data(train_labels==2,:));
    
    [dv, pred] = max(corr(test_data', mean_train')');
    acc(l,i) = sum(pred==test_labels')/length(test_labels);

    end
    
end

%% save RDM
D(l,:,:) = squareform(pdist(imageFeatures));
end
mean_acc = mean(acc,2)