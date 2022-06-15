
load('test.mat') 
num_hid = 20;    
num_out = 1; 
rad      = 10;   % central radius of the half moon
width    = 6;    % width of the half moon
num_samp2 = 1000; % number of samples
d = -7;


T1 = 450;
T2 = 450;
t = 10;
error = zeros(1,t);
for j=1:t
err = 0;
for i = 1:num_samp2  
    x = [nor_data1(1:2,i);-1];    %  3*1
    vhid1 = W1*x;
    yhid1 = mean((kron(vhid1,ones(1,T1))+sigma1.*randn(num_hid,T1))>0,2);    %  20*1
    h_te = [yhid1;-1];
    vout = Wo*h_te;
    y_out(:,i) = mean((kron(vout,ones(1,T2))+sigma2.*randn(num_out,T2))>0,2);   % 1*2000
    xx  = max1(1:2,:).*x(1:2,:) + mean1(1:2,:);   % 2*1 
    if y_out(:,i)>0.5     %myvec2int(o(:,i)) == 1,
        plot(xx(1),xx(2),'rx');
    end
    if y_out(:,i)<0.5     %myvec2int(o(:,i)) == -1,
        plot(xx(1),xx(2),'k+');
    end
end
% Calculate testing error rate
for i = 1:num_samp2
    out(i) = 1*(y_out(i)>=0.5) + 0*(y_out(i)<0.5);
    if abs(out(i) - nor_data1(3,i)) > 1E-6
        err = err + 1;
    end
end
error(j)=(err/num_samp2)*100;
end
mean(error)