function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
h_y = h - y;
h_y_t = h_y';
J1 = h_y_t*h_y;
J2 = theta(2:end)'*theta(2:end);
J  = (1/(2*m))*J1 + (lambda/(2*m))*J2;

grad(1) = (1/m)*(h_y_t*X(:,1));
% grad(2) = (1/m)*(h_y_t*X(:,2)) + (lambda/m)*theta(2);  % WRONG: Features may be > 2 !

% for col = 2 : size(X, 2)
%  grad(col) = (1/m)*(h_y_t*X(:,col)) + (lambda/m)*theta(col);
% end

% Vectorized Version
nel_grad = size(theta, 1); % supposing theta is 1D
subset_grad = [2:nel_grad];
% grad(subset_grad) = (1/m)*X(:,subset_grad)'*h_y + (lambda/m)*theta(subset_grad);
% Alternative Vectorized Version (computationally better, since it transposes less elements)
grad(subset_grad) = ( (1/m)*h_y_t*X(:,subset_grad) )' + (lambda/m)*theta(subset_grad);




% =========================================================================

grad = grad(:);

end
