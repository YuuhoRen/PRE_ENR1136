clc;clear
%处理器：  Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz   3.41 GHz
%操作系统： Windows 10 家庭中文版 64 位操作系统, 基于 x64 的处理器
%MATLAB版本：R2020a
%% 导入据以及定义变量
num_in = 784;     % number of input neuron
num_hid = 100;    % number of hidden neurons
num_out = 10;     % number of output neuron
num_epoch = 100; % number of epochs 300-33  
ENTROPY=zeros(num_epoch,1);
SIGMA=zeros(num_epoch,1);
% SIGMA1=zeros(num_epoch,1);
% SIGMA2=zeros(num_epoch,1);
%% 输入x
Images = loadMNISTImages('E:\mnist-noise\MNIST\t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('E:\mnist-noise\MNIST\t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10


%% 划分数据集
X1 = reshape(Images,784,10000);
% id = randperm(10000);
% train_id = id(:,1:6000);     
% test_id = id(:,6001:8000); 
x_train = X1(:,1:8000);            
d_train = Labels(1:8000);
sigma10 = 1.0;
sigma20 = 1.0;
%% 训练网络
for p=10
    
    sigma1 = sigma10;
    sigma2 = sigma20;
    rng(1024*p)
    W1 = 0.2*rand(num_hid,num_in+1)-0.1;           %输入层到隐藏层的权值
    rng(101*p)
    Wo = 0.2*rand(num_out,num_hid+1)-0.1;            %隐藏层到输出层的权值
    for k=1:1:num_epoch
        alpha = 0.02;%learning rate of weights
        lambda = 0.1;%learning rate of noise level
        beta = 0.1;
        error=0;
        N = length(d_train);
        
        for i = 1:N
            %% Forward computation
            input_tr  = [x_train(1:num_in,i);-1];     % 785*1
            v_hid1 = W1*input_tr;  % 25*785*785*1=25*1
            y_hid1 = qfunc(-v_hid1/sigma1);          % hidden neurons are nonlinear   hk
            h = [y_hid1;-1];
            v_hid2 = Wo*h;    % 10*25*25*1=10*1
            y_out = qfunc(-v_hid2/sigma2);         % output neuron is nonlinear   10*1
            for j = 1:10
                if y_out(j) == 1
                    y_out(j) = y_out(j) - 0.01;
                elseif y_out(j) == 0
                    y_out(j) = y_out(j) + 0.01;
                    
                else
                    y_out(j) = y_out(j);
                end
            end
            d = zeros(10,1);
            d(sub2ind(size(d),d_train(i),1)) = 1;
            e(:,i) = -(d-y_out)./(y_out.*(1-y_out)); % 10*1
            %% Backward computation
            dWo=alpha*e(:,i)*h'.*normpdf(-v_hid2,0,sigma2);   % out*hidden
            dW1=alpha*normpdf(-v_hid2,0,sigma2)'.*Wo(:,1:num_hid)'*e(:,i).*normpdf(-v_hid1,0,sigma1)*input_tr';   % hidden *input
            %% weights update
            Wo = Wo - dWo;  % weights hidden-> output
            W1 = W1 - dW1;  % weights input -> hidden
            error = error-(d'*log(y_out)+(1-d)'*log(1-y_out));
        end
        dsigma2 = e(:,i)'*(-v_hid2.*normpdf(-v_hid2,0,sigma2)/sigma2);
        % dsigma1 = -e(:,i)'*Wo.*normpdf(-v_hid2,0,sigma2)'*W2(:,1:50)*(-v_hid1.*normpdf(-v_hid1,0,sigma1)/sigma1);
        dsigma1 = e(:,i)'*(Wo(:,1:num_hid).*normpdf(-v_hid2,0,sigma2)*(-v_hid1.*normpdf(-v_hid1,0,sigma1)/sigma1));
        % dsigma=-e(:,i)'*Wo*(-v_hid.*normpdf(-v_hid,0,sigma)/sigma)+2*beta*sigma;   %1*25*25*1
        sigma2 = sigma2 - lambda*dsigma2;
        sigma1 = sigma1 - lambda*dsigma1;
        SIGMA2(k) = sigma2;
        SIGMA1(k) = sigma1;
        %      if sigma1 && sigma2<=lambda
        %          break
        %      end
        ENTROPY(k)=error;
        sprintf('第%d次迭代，误差为%f',k,ENTROPY(k))
    end
    ENTROPY(end);
end
figure(1);
semilogy(ENTROPY,'b');
% plot(log10(ENTROPY),'k');
title('training set error');
xlabel('number of epochs','fontsize',15);
ylabel('total cross entropy ','fontsize',15);
legend('training set error');  %加上图例
%set(gca,'XLim',[0 10000]);
figure(2)
plot([0:1:num_epoch],[sigma10,SIGMA1],'-b*')
xlabel('epoch','fontsize',15) 
ylabel('\sigma','fontsize',15)
hold on
plot([0:1:num_epoch],[sigma20,SIGMA2],'-mv')
xlabel('epoch','fontsize',15) 
ylabel('\sigma','fontsize',15)


acc = 0;
T1 = 400;
T2 = 400;
Images_noise = Images;
% Images_noise([7:14],[7:14],:) = 1; % 62.25
% Images_noise = imnoise(Images, 'speckle',0.3);% 94.6
%   Images_noise = imnoise(Images, 'gaussian',0,0.1); %71.9
Images_noise = imnoise(Images,'salt & pepper',0.1); % 91.65(0.1) 77.05(0.2)
figure(3)
imshow(Images_noise(:,:,2124));
figure(4)
imshow(Images(:,:,2124));
X2 = reshape(Images_noise,784,10000);
x_test = X2(:,8001:10000);
d_test = Labels(8001:10000);
M = length(d_test);
for k=1:M
    input_1=[x_test(:,k);-1];
    v_hid_1= W1*input_1;  %   50*1
    y_hid_1 = mean((kron(v_hid_1,ones(1,T1))+sigma1*randn(num_hid,T1))>0,2);      %计算隐藏单元的输出  
    v_out_1 = Wo*[y_hid_1;-1];      %
    y_out_1 = mean((kron(v_out_1,ones(1,T2))+sigma2*randn(num_out,T2))>0,2);
    [~,i] = max(y_out_1);
    if i == d_test(k)
        acc = acc + 1;
    end
end
acc = acc/M;
fprintf('Accuracy is %f\n',acc);



