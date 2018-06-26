% code to categorize stimuli based on image features 
% im_features can be set to pixel intensity (RGB or grayscale) or output of
% different VGG layers (set layers below)
clear all
im_dir =  '~/Dropbox (MIT)/Projects/social_interaction/stimuli/MEG_stills/CNN_experiment/';
% 2 classes of images, in separate directories
class1_dir = [im_dir '/interaction/'];
class2_dir = [im_dir '/non-interaction/'];

im_features = 'vgg'; %'vgg32'; %'gray'; %options: grayscale, rgb, vgg  
vgg_layers = [6, 11, 18, 25, 32];% 5 pooling layers in VGG are 06, 11, 18, 25, 32
im_format = '.jpg';
run_pca = 1;
num_PCs = 40;


% number of SVM resample runs
resample_runs = 20;

class1 = dir([class1_dir '*' im_format]);
class1 = {class1.name};
class2 = dir([class2_dir '*' im_format]);
class2 = {class2.name};

if strcmp(im_features, 'vgg')
    % can modify here with AlexNet if preferred
    net = vgg16;
    imageSize = net.Layers(1).InputSize;
    layers = vgg_layers;
else
    layers = 1; %dummy variable for pixel case
end
acc= zeros(length(layers),resample_runs);
D = zeros(length(layers), length(class1)*2, length(class1)*2);

for l = 1:length(layers)
    clear class1_ims class2_ims
for i = 1:length(class1)
    
    if strcmpi(im_features, 'gray')
    tmp1 = rgb2gray(imread([class1_dir class1{i}]));
    tmp2 = rgb2gray(imread([class2_dir class2{i}]));
    
    elseif strcmpi(im_features, 'rgb')
    tmp1 = imread([class1_dir class1{i}]);
    tmp2 = imread([class2_dir class2{i}]);
    
    elseif strcmpi(im_features, 'VGG')
    layer = layers(l);
  
    tmp1 = imread([class1_dir class1{i}]);
    tmp1 = augmentedImageDatastore(imageSize, tmp1, 'ColorPreprocessing', 'gray2rgb');
    tmp1 = activations(net, tmp1, layer, 'OutputAs', 'columns');
    
    tmp2 = imread([class2_dir class2{i}]);
    tmp2 = augmentedImageDatastore(imageSize, tmp2, 'ColorPreprocessing', 'gray2rgb');
    tmp2 = activations(net, tmp2, layer, 'OutputAs', 'columns');
   
    else
    error('Not a valid image feature \n options are rgb, gray, VGGNN (where NN is layer #')
    end
    
    class1_ims(i,:) = tmp1(:);
    class2_ims(i,:) = tmp2(:);
end

imageFeatures = double([class1_ims; class2_ims]);
labels = [ones(1,length(class1)), 2*ones(1,length(class2))];

if run_pca
imageFeatures_orig = imageFeatures;
[coeff,imageFeatures_pca,latent,tsquared,explained,mu] = pca(imageFeatures); 
imageFeatures = imageFeatures_pca(:,1:num_PCs);
end

for i = 1:resample_runs
    % 80-20 crossvalidation split
    [train,test] = crossvalind('holdout',labels,0.2);
    
    train_labels = labels(train);
    test_labels = labels(test)';
    train_data = imageFeatures(train,:);
    test_data = imageFeatures(test,:);
    
    SVMStruct = fitcsvm(train_data,train_labels);
    pred = predict(SVMStruct, test_data);
    acc(l,i) = sum(pred==test_labels)/length(test_labels);
    
end

%% save RDM
D(l,:,:) = squareform(pdist(imageFeatures));
end
mean_acc = mean(acc,2)