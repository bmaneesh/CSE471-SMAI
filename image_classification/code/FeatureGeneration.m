clc
clear all
close all

%% Load Data and Generate Training and Testing Data
%   1:MotorBike       2:AirPlane      3:Face      4:Watch
fprintf('Loading Data....\n')
load Bikes.mat;
load AirPlane.mat;
load Background.mat;
load Face.mat;
for i=1:length(MB)
    Data{i} =  MB{i};
    Label(i) = 1;
end
m=i;
for i=i+1:length(MB)+length(AP)
    Data{i} =  AP{i-m};
    Label(i) = 2;
end
m=i;
for i=i+1:length(MB)+length(AP)+length(FC)
    Data{i} =  FC{i-m};
    Label(i) = 3;
end
m=i;
for i=i+1:length(MB)+length(AP)+length(FC)+length(car)
    Data{i} =  car{i-m};
    Label(i) = 4;
end
LN = length(Data); % Total Dataset length
clear MB; clear AP;clear FC;

Training = Data;
Training_L = Label;


NoofTrain = length(Training_L);

%% SIFT Features
fprintf('Finding SIFT Features....\n')
data = [];
for i=1:NoofTrain
    clc
    fprintf('Loading Data....\n')
    fprintf('Finding SIFT Features.... %f \n',i*100/NoofTrain)
    [f{i},d{i}] = vl_sift(single(Training{i}));       % Compute SIFT Features
    data = [data  single(d{i})];
end

%% K-Means Clustering
fprintf('Running K-Means Clustering....\n')
k = 5000;     % Number of Clusters
[centers, assignments] = vl_kmeans(data,k);

%% Histogram Representation
fprintf('Generating Histogram Features....\n')
Hist = zeros(k,NoofTrain);
x=1;y=0;    % i-th image feature starting point and ending point
for i=1:NoofTrain  % for each image
    x = y+1;
    y = x + size(d{i},2)-1;
    ass = assignments(x:y);
    Hist(:,i) = histc(ass,1:k)';
end