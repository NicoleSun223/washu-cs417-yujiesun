function [num_iters, bounds_minus_n] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)
i=1;
num_iters=zeros(1,num_samples);
bounds=zeros(1,num_samples);
bounds_minus_n=zeros(1,num_samples);
log_differences=zeros(1,num_samples);
while i <= num_samples
    x=unifrnd(-1,1,[N,d+1]);
    %Set the first column to be 1
    x(:,1)=1;
    v=zeros(1,1);
    u=unifrnd(0,1,[1,d]);
    w_star=[v,u];
    y=sign(w_star*x');
    %Compute bounds
    r=[];
    for j=1:N
    r=[r,norm(x(j,:))];
    end   
    r_square=(max(r))^2;
    p_square=(min(y.*(w_star*x')))^2;
    bounds(i)=r_square*(norm(w_star))^2/p_square;
    bounds_minus_n(i)=bounds(i)-num_iters(i);
    log_differences(i)=log(bounds_minus_n(i));
    
    dataset=[x,y'];
    [~, num_iters(i)] = perceptron_learn(dataset);
    i=i+1;
end

figure(1)
histogram(num_iters,'BinLimits',[0,1500])
xlabel('Iterations')
ylabel('Frequency')
legend('Number of Iterations')

figure(2)
histogram(log_differences,'BinLimits',[7,25])
xlabel('log of the differences between bounds and number of iterations')
ylabel('Frequency')
legend('log of the differences')
end


    