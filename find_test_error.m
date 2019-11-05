function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 
[a,~] = size(X);
X1 = ones(a,1);
X = [X1,X];
test_error = 1/2*(sum(abs((sign(exp(w'*X')./(1+exp(w'*X'))-0.5))'-y))/a);
end

