function [result,alpha,b] = lssvrfit_pso_sp(X,Y,gam,sig2,Xtest,Ytest)

% train LSSVR
type = 'function estimation';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});
% test
Ypredict = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
% evaluate performance 
% result = mse(Ypredict-Ytest);
result = sqrt(mse(Ypredict-Ytest));
