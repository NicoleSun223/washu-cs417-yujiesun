function [t, w, e_in, time] = logistic_reg(X, y, w_init, max_its, eta)
% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%     
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error 

%Before Scaling
%D=[ones(size(X,1),1),X];
%w=w_init;
%tic
%for t=0:max_its
%    g=zeros(size(D));
%    for i=1:size(D,1)
%       g(i,:)=y(i).*X1(i,:)/(1+exp(y(i)*(w'*D(i,:)')));
%    end
%    gradient=-mean(g);
%    w=w+eta*(-gradient');
%    if max(abs(gradient))<10^-3
%        break
%    end
%end
%Compute E_in
%e_in=mean(log(1+exp(-y.*(D*w))));
%time = toc;

%After Scaling
X_scaling=zscore(X);
D=[ones(size(X_scaling,1),1),X_scaling];
w=w_init;
tic
%Set Max_its to be a extremly large number
for t=0:max_its
    g_temp=zeros(size(D));
    for i=1:size(D,1)
        g_temp(i,:)=y(i).*D(i,:)/(1+exp(y(i)*(w'*D(i,:)')));
    end
    gradient=-mean(g_temp);
    w=w+eta*(-gradient');
    if max(abs(gradient))<=10^-6
        break
    end
end
%Compute E_in
e_in=mean(log(1+exp(-y.*(D*w))));
time=toc;
end

