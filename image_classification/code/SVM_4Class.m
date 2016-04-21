clc
clear all
close all

%% SVM: One Vs All
%   1:MotorBike       2:AirPlane      3:Face      4:Watch
% load SVM_4ClassData.mat;
load Data2_50.mat
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

%% SVM 1: Motorbike vs Airplane
[w1,b1,info1] = vl_svmtrain([Training1 Training2], [Training_L1 Training_L2-3], lambda, 'MaxNumIterations', maxIter);
[sum(((w1'*[Testing1 Testing2]+b1)>0)==(([Testing_L1 Testing_L2-3])>0))*100/(2*length(Testing_L1)) sum(((w1'*[Training1 Training2]+b1)>0)==([Training_L1 Training_L2-3]>0))*100/(2*length(Training_L1))]

%% SVM 2: Motorbike vs Face
[w2,b2,info2] = vl_svmtrain([Training1 Training3], [Training_L1 Training_L3-4], lambda, 'MaxNumIterations', maxIter);
[sum(((w2'*[Testing1 Testing3]+b2)>0)==(([Testing_L1 Testing_L3-4])>0))*100/(2*length(Testing_L1)) sum(((w2'*[Training1 Training3]+b2)>0)==([Training_L1 Training_L3-4]>0))*100/(2*length(Training_L1))]

%% SVM 3: Motorbike vs Watch
[w3,b3,info3] = vl_svmtrain([Training1 Training4], [Training_L1 Training_L4-5], lambda, 'MaxNumIterations', maxIter);
[sum(((w3'*[Testing1 Testing4]+b3)>0)==(([Testing_L1 Testing_L4-5])>0))*100/(2*length(Testing_L1)) sum(((w3'*[Training1 Training4]+b3)>0)==([Training_L1 Training_L4-5]>0))*100/(2*length(Training_L1))]

%% SVM 4: AirPlane vs Face
[w4,b4,info4] = vl_svmtrain([Training2 Training3], [Training_L2-1 Training_L3-4], lambda, 'MaxNumIterations', maxIter);
[sum(((w4'*[Testing2 Testing3]+b4)>0)==(([Testing_L2-1 Testing_L3-4])>0))*100/(2*length(Testing_L1)) sum(((w4'*[Training2 Training3]+b4)>0)==([Training_L2-1 Training_L3-4]>0))*100/(2*length(Training_L1))]

%% SVM 5: AirPlane vs Watch
[w5,b5,info5] = vl_svmtrain([Training2 Training4], [Training_L2-1 Training_L4-5], lambda, 'MaxNumIterations', maxIter);
[sum(((w5'*[Testing2 Testing4]+b5)>0)==(([Testing_L2-1 Testing_L4-5])>0))*100/(2*length(Testing_L1)) sum(((w5'*[Training2 Training4]+b5)>0)==([Training_L2-1 Training_L4-5]>0))*100/(2*length(Training_L1))]

%% SVM 6: Face vs Watch
[w6,b6,info6] = vl_svmtrain([Training3 Training4], [Training_L3-2 Training_L4-5], lambda, 'MaxNumIterations', maxIter);
[sum(((w6'*[Testing3 Testing4]+b6)>0)==(([Testing_L3-2 Testing_L4-5])>0))*100/(2*length(Testing_L1)) sum(((w6'*[Training3 Training4]+b6)>0)==([Training_L3-2 Training_L4-5]>0))*100/(2*length(Training_L1))]


%% Testing...
Tst = [Testing1 Testing2 Testing3 Testing4];
Tst_L = [Testing_L1 Testing_L2 Testing_L3 Testing_L4];
Cuml = zeros(4,35*4);

Cuml(1,:) = Cuml(1,:) + ((w1'*Tst+b1)>0);
Cuml(2,:) = Cuml(2,:) + ((w1'*Tst+b1)<0);
Cuml(1,:) = Cuml(1,:) + ((w2'*Tst+b2)>0);
Cuml(3,:) = Cuml(3,:) + ((w2'*Tst+b2)<0);
Cuml(1,:) = Cuml(1,:) + ((w3'*Tst+b3)>0);
Cuml(4,:) = Cuml(4,:) + ((w3'*Tst+b3)<0);
Cuml(2,:) = Cuml(2,:) + ((w4'*Tst+b4)>0);
Cuml(3,:) = Cuml(3,:) + ((w4'*Tst+b4)<0);
Cuml(2,:) = Cuml(2,:) + ((w5'*Tst+b5)>0);
Cuml(4,:) = Cuml(4,:) + ((w5'*Tst+b5)<0);
Cuml(3,:) = Cuml(3,:) + ((w6'*Tst+b6)>0);
Cuml(4,:) = Cuml(4,:) + ((w6'*Tst+b6)<0);

[Y, I] = max(Cuml);
fprintf('Final Accuracy: %f percentage\n',sum(I==Tst_L)*100/length(Tst_L));
% [Cuml' I' Tst_L']
% fprintf('Final Accuracy: %f percentage\n',sum(I==Tst_L)*100/length(Tst_L));

round([sum(I(35*0+(1:35))==1)*100/35 sum(I(35*0+(1:35))==2)*100/35 sum(I(35*0+(1:35))==3)*100/35 sum(I(35*0+(1:35))==4)*100/35;
 sum(I(35*1+(1:35))==1)*100/35 sum(I(35*1+(1:35))==2)*100/35 sum(I(35*1+(1:35))==3)*100/35 sum(I(35*1+(1:35))==4)*100/35;
 sum(I(35*2+(1:35))==1)*100/35 sum(I(35*2+(1:35))==2)*100/35 sum(I(35*2+(1:35))==3)*100/35 sum(I(35*2+(1:35))==4)*100/35;
 sum(I(35*3+(1:35))==1)*100/35 sum(I(35*3+(1:35))==2)*100/35 sum(I(35*3+(1:35))==3)*100/35 sum(I(35*3+(1:35))==4)*100/35])