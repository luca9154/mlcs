function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



% (NOT GOOD)
% THIS VECTORIZED VECTOR HAS MANY ERRORS
% REGARDING COMBINATIONS OF Is and Js.
% ===
%R
%[i, j]  = find(R == 1);
%thetaJs = Theta(j,:)
%XIs     = X(i,:)
%pause
%
%% rated     = find(R == 1)
%Y
%Ys        = zeros(size(i, 1), size(j, 1))
%% Ys(rated) = Y(rated)
%Ys = Y(i,j)
%pause
%
%J = sum( sum( ( XIs*thetaJs' - Ys  ).^2 ) ) / 2. ;


% (GOOD)
% NON-VECTORIZED VERSION
% ===
%[i, j] = find(R == 1);
%n = size(i, 1);
%for k = 1 : n
%  a = i(k); b = j(k);
%  theta = Theta(b,:);
%  x = X(a,:)';
%  J = J + ( ( theta*x - Y(a,b) )^2 )/2.;
%end


% (GOOD)
% VECTORIZED VERSION
% ===
multt = (X*(Theta)' - Y).*R;  % this may be bad if a lot of movies have not been watched yet
only1 = (multt).^2;  % set to 0 every element corresponding to movies without rating
J = sum(sum(only1))/2.;

X_grad = multt*Theta;
Theta_grad = (multt')*X;

% REGULARIZATION
% ===
regTheta = sum(sum( (Theta.^2) ))*lambda*0.5;
regX     = sum(sum( (X.^2) ))*lambda*0.5;
J = J + regTheta + regX;

X_grad = X_grad + lambda*X;
Theta_grad = Theta_grad + lambda*Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
