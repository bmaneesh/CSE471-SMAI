clc
clear all
close all

%% SVM: One Vs All
%   1:MotorBike       2:AirPlane      3:Face      4:Watch
% load SVM_4ClassData.mat;
load Data2_5000.mat
lambda = 1 ; % Regularization parameter
maxIter = 10000 ; % Maximum number of iterations
sel = randperm(LN/4);
% Hist = Hist ./ repmat(sum(Hist),[50,1]);
Training1 = Hist(:,435*0+sel(1:400));
Training2 = Hist(:,435*1+sel(1:400)) ;
Training3 = Hist(:,435*2+sel(1:400));
Training4 = Hist(:,435*3+sel(1:400));

Training_L1 = ones(1,400)*1;
Training_L2 = ones(1,400)*2;
Training_L3 = ones(1,400)*3;
Training_L4 = ones(1,400)*4;

Testing1 = Hist(:,435*0+sel(400+1:435));
Testing2 = Hist(:,435*1+sel(400+1:435));
Testing3 = Hist(:,435*2+sel(400+1:435));
Testing4 = Hist(:,435*3+sel(400+1:435));

Testing_L1 = ones(1,35)*1;
Testing_L2 = ones(1,35)*2;
Testing_L3 = ones(1,35)*3;
Testing_L4 = ones(1,35)*4;

Training = [Training1 Training2 Training3 Training4]; 
Training_L = [Training_L1 Training_L2 Training_L3 Training_L4];
Testing = [Testing1 Testing2 Testing3 Testing4];
Testing_L = [Testing_L1 Testing_L2 Testing_L3 Testing_L4];
for i=1:length(Training_L)
    label(Training_L(i),i)=1;
end

for i=1:length(Testing_L)
    label_L(Testing_L(i),i)=1;
end