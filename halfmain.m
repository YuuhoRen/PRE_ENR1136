%=========== Generating halfmoon data ============================
rad      = 10;   % central radius of the half moon
width    = 6;    % width of the half moon
num_train   = 1000;   % number of training sets
num_test   = 508;    % number of testing sets
num_samp = num_train+num_test; % number of samples
dist = -7;
[data, data_shuffled] = SRA_halfmoon(rad,width,dist,num_samp);

%手动添加噪声数据点
%蓝色区域的红色点
r1=[-3,10,0];r2=[1,7,0];r3=[4,5.5,0];r4=[7,2,0];
data_shuffled=[data_shuffled(:,1:1050) r1' data_shuffled(:,1051:1501)];
data_shuffled=[data_shuffled(:,1:1123) r2' data_shuffled(:,1124:1502)];
data_shuffled=[data_shuffled(:,1:1338) r3' data_shuffled(:,1339:1503)];
data_shuffled=[data_shuffled(:,1:1221) r4' data_shuffled(:,1222:1504)];
%红色区域的蓝色点
b1=[2,2,1];b2=[5,0,1];b3=[10,-2,1];b4=[16,0,1];
data_shuffled=[data_shuffled(:,1:1199) b1' data_shuffled(:,1200:1505)];
data_shuffled=[data_shuffled(:,1:1388) b2' data_shuffled(:,1389:1506)];
data_shuffled=[data_shuffled(:,1:1321) b3' data_shuffled(:,1322:1507)];
data_shuffled=[data_shuffled(:,1:1279) b4' data_shuffled(:,1280:1508)];

%%============ network structure=====================================
num_in = 2;     % number of input neuron
num_hid = 20;    % number of hidden neurons
num_out = 1;     % number of output neuron
num_epoch =100; % number of epochs
mse_train = Inf;     % MSE for training data
err=0;
ENTROPY=zeros(num_epoch,1);
SIGMA1=zeros(num_epoch,1);
SIGMA2=zeros(num_epoch,1);
%%========= Preprocess the input data : remove mean and normalize =========
mean1 = [mean(data(1:2,:)')';0];    % 3*1   x_mean、y_mean、0
for i = 1:num_samp
    nor_data(:,i) = data_shuffled(:,i) - mean1;  % 3*3000  
end
max1  = [max(abs(nor_data(1:2,:)'))';1];   % 3*1  x_max、y_max、1
for i = 1:num_samp
    nor_data(:,i) = nor_data(:,i)./max1;   % normalize the input 3*3000
end
%%======================= Main Loop for Training ==========================
st = cputime;
fprintf('Training the MLP using back-propagation ...\n');
fprintf('  ------------------------------------\n');
for p=11
    sigma1 = 0.8;
    sigma2 = 0.8;
    rng(1024*p)
    W1 = 0.2*rand(num_hid,num_in+1)-0.1;   %  20*3
    rng(101*p)
    Wo = 0.2*rand(num_out,num_hid+1)-0.1;    % 1*20
 for k=1:1:num_epoch
    [n_row, n_col] = size(nor_data); %3*3000
    shuffle_seq = randperm(num_train);   % 1-1000数字打乱重排
    nor_data1 = nor_data(:,shuffle_seq);  %3*1000  
    alpha = 0.1;%learning rate of weights     0.1损失=0.0309 0.02=0.1288
    lambda = 0.1;%learning rate of noise level
    error=0;
    for i = 1:num_train
        %% Forward computation
        x  = [nor_data1(1:2,i);-1];     % fetching input data from database从数据库中输入数据  3*1
        d  = nor_data1(3,i);% fetching desired response from database
        v_hid1=W1*x;  % 20*3*3*1
        y_hid1 = qfunc(-v_hid1/sigma1);          % hidden neurons are nonlinear  20*1
        h_tr = [y_hid1;-1];
        v_out = Wo*h_tr;    % 1*1
        y_out = qfunc(-v_out/sigma2);         % output neuron is nonlinear   
            if y_out == 1 
                y_out = y_out - 0.01;
            elseif y_out == 0
                    y_out = y_out + 0.01;
            else
                    y_out = y_out; 
            end
        e(:,i) = -(d-y_out)./(y_out.*(1-y_out)); % 1*1
        
        %% Backward computation
        dWo = alpha*e(:,i)*h_tr'.*normpdf(-v_out,0,sigma2);   %1*21
        dW1 = alpha*normpdf(-v_out,0,sigma2)'.*Wo(:,1:num_hid)'*e(:,i).*normpdf(-v_hid1,0,sigma1)*x';
%       dW1=alpha*(-Wo')*e(:,i).*normpdf(-v_hid1,0,sigma)*x';   %20*1*1*3 
        %% weights update
        Wo = Wo - dWo;  % weights hidden-> output  
        W1 = W1 - dW1;  % weights input -> hidden
%         cross_entropy=-(d*log(y_out)+(1-d)*log(1-y_out));
        error=error-(d*log(y_out)+(1-d)*log(1-y_out));%total cross entropy
    
    end
    dsigma2 = e(:,i)'*(-v_out.*normpdf(-v_out,0,sigma2)/sigma2);
    dsigma1 = e(:,i)'*(Wo(:,1:num_hid).*normpdf(-v_out,0,sigma2)*(-v_hid1.*normpdf(-v_hid1,0,sigma1)/sigma1));
    % dsigma1 = e(:,i)'*Wo*(-v_hid1.*normpdf(-v_hid1,0,sigma)/sigma);   %1*20*20*1
        sigma2=sigma2-lambda*dsigma2;
        sigma1=sigma1-lambda*dsigma1;
        SIGMA1(k)=sigma1;
        SIGMA2(k)=sigma2;
%      if sigma1 && sigma2<=lambda
%          break
%      end
     ENTROPY(k) =error;
     mse_train = ENTROPY(k);
 end  
  ENTROPY(end);
%      if ENTROPY(end)<10^-2 && SIGMA(end-100)>=SIGMA(end)
%         break
%      end
end
fprintf('   Points trained : %d\n',num_train);
fprintf('  Epochs conducted: %d\n',num_epoch);
fprintf('        Time cost : %4.2f seconds\n',cputime - st);
fprintf('  ------------------------------------\n');

% figure(1);
% semilogy(ENTROPY,'b');
% title('Learning curve');
% xlabel('Number of epochs','fontsize',15);
% ylabel('total cross-entropy error','fontsize',15);
% %set(gca,'XLim',[0 10000]);
% figure(2)
% plot([0:num_epoch],[0.8;SIGMA1],'b*-')
% xlabel('epoch','fontsize',15) 
% ylabel('\sigma','fontsize',15)
% figure(3)
% plot([0:num_epoch],[0.8;SIGMA2],'ro-')
% xlabel('epoch','fontsize',15) 
% ylabel('\sigma','fontsize',15)

% figure;
% hold on;
xmin = min(data_shuffled(1,:));
xmax = max(data_shuffled(1,:));
ymin = min(data_shuffled(2,:));
ymax = max(data_shuffled(2,:));
[x_b,y_b]= meshgrid(xmin:(xmax-xmin)/100:xmax,ymin:(ymax-ymin)/100:ymax);   % 101*101
z_b  = 0*ones(size(x_b));
%wh = waitbar(0,'Plotting testing result...');
for x1 = 1 : size(x_b,1)  % x1=101
    for y1 = 1 : size(x_b,2)    % y1=101
        input = [(x_b(x1,y1)-mean1(1))/max1(1);(y_b(x1,y1)-mean1(2))/max1(2);-1];   % 3*1 从x_b的行开始输入数据
         y_hid1 = [qfunc(-W1*input/sigma1);-1];    %20*1
        z_b(x1,y1) = qfunc((-Wo*y_hid1)/sigma2);
    end
    %waitbar((x1)/size(x,1),wh)
    %set(wh,'name',['Progress = ' sprintf('%2.1f',(x1)/size(x,1)*100) '%']);
end

%figure;
% sp = pcolor(x_b,y_b,z_b);   % 绘制伪彩色图
% load red_black_colmap;    %256*3
% colormap(red_black);
% shading flat;  % 去掉网格线
% set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);    %  设定图形的显示范围

%%========================== Testing ======================================
fprintf('Testing the MLP when dist \n');
T1 = 500;
T2 = 500;
% for i = num_train+1:num_samp    % 从第1001个数据到第3000个数据
%     x = [nor_data(1:2,i);-1];    %  3*1
%     vhid1 = W1*x;
%     yhid1 = mean((kron(vhid1,ones(1,T1))+sigma1*randn(num_hid,T1))>0,2);    %  20*1
%     h_te = [yhid1;-1];
%     vout = Wo*h_te;
%     y_out(:,i) = mean((kron(vout,ones(1,T2))+sigma2*randn(num_out,T2))>0,2);   % 1*2000
%     xx  = max1(1:2,:).*x(1:2,:) + mean1(1:2,:);   % 2*1 
%     if y_out(:,i)>0.5     %myvec2int(o(:,i)) == 1,
%         plot(xx(1),xx(2),'rx');
%     end
%     if y_out(:,i)<0.5     %myvec2int(o(:,i)) == -1,
%         plot(xx(1),xx(2),'k+');
%     end
% end
% xlabel('x');ylabel('y');
% title(['Classification using MLP with dist = ',num2str(dist), ', radius = ',...
%        num2str(rad), ' and width = ',num2str(width)]);
% % Calculate testing error rate
% for i = num_train+1:num_samp
%     out(i) = 1*(y_out(i)>=0.5) + 0*(y_out(i)<0.5);
%     if abs(out(i) - nor_data(3,i)) > 1E-6
%         err = err + 1;
%     end
% end
% fprintf('  ------------------------------------\n');
% fprintf('   Points tested : %d\n',num_test);
% fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_test)*100);
% fprintf('  ------------------------------------\n');
% %%======================= Plot decision boundary ==========================
% %% Adding contour to show the boundary
% contour(x_b,y_b,z_b,[1/2 1/2],'k','Linewidth',1);    % 画出值为的等值线
% %contour(x_b,y_b,z_b,[-1 -1],'k:','Linewidth',2);
% %contour(x_b,y_b,z_b,[1 1],'k:','Linewidth',2);
% set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);


figure
hold on
for i = num_train+1:num_samp    % 从第1001个数据到第3000个数据
    x = [nor_data(1:2,i);-1];    %  3*1
    vhid1 = W1*x;
    yhid1 = mean((kron(vhid1,ones(1,T1))+sigma1*randn(num_hid,T1))>0,2);    %  20*1
    h_te = [yhid1;-1];
    vout = Wo*h_te;
    y_out(:,i) = mean((kron(vout,ones(1,T2))+sigma2*randn(num_out,T2))>0,2);   % 1*2000
    xx  = max1(1:2,:).*x(1:2,:) + mean1(1:2,:);   % 2*1 
    if y_out(:,i)>0.5     %myvec2int(o(:,i)) == 1,
        plot(xx(1),xx(2),'bo','MarkerSize',10,'LineWidth',1)
    end
    if y_out(:,i)<0.5     %myvec2int(o(:,i)) == -1,
        plot(xx(1),xx(2),'rx','MarkerSize',10,'LineWidth',1);
    end 
end

plot(-3,10,'rx','MarkerSize',10,'LineWidth',1)
plot(1,7,'rx','MarkerSize',10,'LineWidth',1)
plot(4,5.5,'rx','MarkerSize',10,'LineWidth',1)
plot(7,2,'rx','MarkerSize',10,'LineWidth',1)

plot(2,2,'bo','MarkerSize',10,'LineWidth',1)
plot(5,0,'bo','MarkerSize',10,'LineWidth',1)
plot(10,-2,'bo','MarkerSize',10,'LineWidth',1)
plot(16,0,'bo','MarkerSize',10,'LineWidth',1)

xlabel('x');ylabel('y');
title(['Classification using MLP with dist = ',num2str(dist), ', radius = ',...
       num2str(rad), ' and width = ',num2str(width)]);
% Calculate testing error rate
for i = num_train+1:num_samp
    out(i) = 1*(y_out(i)>=0.5) + 0*(y_out(i)<0.5);
    if abs(out(i) - nor_data(3,i)) > 1E-6
        err = err + 1;
    end
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',num_test);
fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_test)*100);
fprintf('  ------------------------------------\n');
contour(x_b,y_b,z_b,[1/2 1/2],'k','Linewidth',1);    % 画出值为的等值线
%contour(x_b,y_b,z_b,[-1 -1],'k:','Linewidth',2);
%contour(x_b,y_b,z_b,[1 1],'k:','Linewidth',2);
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
save('test.mat','W1','Wo','nor_data','sigma1','sigma2')