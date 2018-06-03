im_dir =  '~/Dropbox (MIT)/Projects/social_interaction/stimuli/MEG_stills/CNN_experiment/';
im_format = '.jpg';
% 2 classes of images, in separate directories
class1_dir = [im_dir '/interaction/'];
class2_dir = [im_dir '/non-interaction/'];
im_size = [1080 1920];

% number of SVM resample runs
resample_runs = 20;

class1 = dir([class1_dir '*' im_format]);
class1 = {class1.name};
class2 = dir([class2_dir '*' im_format]);
class2 = {class2.name};

class1_ims = zeros(length(class1), im_size(1)*im_size(2));
class2_ims = zeros(length(class1), im_size(1)*im_size(2));


for i = 1:length(class1)
    tmp = rgb2gray(imread([class1_dir class1{i}]));
    class1_ims(i,:) = tmp(:);

    tmp = rgb2gray(imread([class2_dir class2{i}]));
    class2_ims(i,:) = tmp(:);
end

imageFeatures = double([class1_ims; class2_ims]);
labels = [ones(1,length(class1)), 2*ones(1,length(class2))];
acc= zeros(1,resample_runs);

for i = 1:resample_runs
    % 80-20 crossvalidation split
    [train,test] = crossvalind('holdout',labels,0.2);
    
    train_labels = labels(train);
    test_labels = labels(test)';
    train_data = imageFeatures(train,:);
    test_data = imageFeatures(test,:);
    
    SVMStruct = fitcsvm(train_data,train_labels);
    pred = predict(SVMStruct, test_data);
    acc(i) = sum(pred==test_labels)/length(test_labels);
    
end

mean_acc = mean(acc)
