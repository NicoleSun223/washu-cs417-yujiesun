%Import training data
train='cleveland_train.csv';
train_set=csvread(train,1,0 );
X_train=train_set(:,1:end-1);
y_train=train_set(:,end);
y_train=2*y_train-1;
%Import test data
test='cleveland_test.csv';
test_set=csvread(test,1,0 );
X_test=test_set(:,1:end-1);
y_test=test_set(:,end);
y_test=2*y_test-1;
%Feature scaling
X_train_scaling=zscore(X_train);
X_test_scaling=zscore(X_test);
%Initial weight
w_initial=zeros(14,1);

X1 = [ones(size(X_train,1),1),X_train];
eta=10^-5;
%max_its=10^6;
%[t,w,e_in,time]=logistic_reg(X_train,y_train,w_initial,max_its, eta);
%[error_train] = find_test_error(w,X_train,y_train );
%[error_test] = find_test_error(w,X_test,y_test );
%result=[e_in,error_train,error_test,time];
%disp(result);
max_its2=10^4;
eta2=10^-5;
[t,w,e_in,time]=logistic_reg(X_train_scaling,y_train,w_initial,max_its2, eta2);
[error_train] = find_test_error(w,X_train_scaling,y_train );
[error_test] = find_test_error(w,X_test_scaling,y_test );
result=[t,e_in,error_train,error_test,time];
disp(result);
%result_10k = [e_in; num_its; execution_time; clss_error_train; test_error_test];
