function [err] = computeTestError(X_poly, y, X_poly_test, ytest, lambda)
  theta = trainLinearReg(X_poly, y, lambda); % train (with training set!)
  err   = linearRegCostFunction(X_poly_test, ytest, theta, 0); % test error (with test set, lambda = 0)
end
