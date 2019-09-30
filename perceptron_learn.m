function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%iterations is the number of iterations the algorithm ran for
[n,k]=size(data_in);
x=data_in(:,1:end-1);
y=data_in(:,end);
w=zeros(1,k-1);
condition=1;
count=0;
while condition==1
    condition=0;
    for i=1:n
        if y(i)~=sign(w*x(i,1:k-1)')
            count=count+1;
            w=w+y(i)*x(i,1:k-1);
            condition=1;
        end
    end
end
iterations=count;
end
    
    