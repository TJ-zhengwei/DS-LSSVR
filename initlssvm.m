function model = initlssvm(X,Y,type, gam,sig2, kernel_type, preprocess)
% Initiate the object oriented structure representing the LS-SVM model
%
%   model = initlssvm(X,Y, type, gam, sig2)
%   model = initlssvm(X,Y, type, gam, sig2, kernel_type)
%
% Full syntax
% 
% >> model = initlssvm(X, Y, type, gam, sig2, kernel, preprocess)
% 
%       Outputs    
%         model         : Object oriented representation of the LS-SVM model
%       Inputs    
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         kernel(*)     : Kernel type (by default 'RBF_kernel')
%         preprocess(*) : 'preprocess'(*) or 'original' 
%
% see also:
%   trainlssvm, simlssvm, changelssvm, codelssvm, prelssvm

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab



% check enough arguments?
if nargin<5,
  error('Not enough arguments to initialize model..');
% isnumeric()判断输入参数是否是数字类型（包括浮点型和整型）
elseif ~isnumeric(sig2),
  error(['Kernel parameter ''sig2''needs to be a (array of) reals' ...
	 'or the empty matrix..']); 
end

%
% CHECK TYPE
% At present, the 'type' only supports 'f' and 'c'. 2017-1-22
%
if type(1)~='f'
    if type(1)~='c'
        if type(1)~='t'
            if type(1)~='N'
                error('type has to be ''function (estimation)'', ''classification'', ''timeserie'' or ''NARX''');
            end
        end
    end
end
model.type = type;

%
% check datapoints
%
model.x_dim = size(X,2);
model.y_dim = size(Y,2);

if and(type(1)~='t',and(size(X,1)~=size(Y,1),size(X,2)~=0)), error('number of datapoints not equal to number of targetpoints...'); end  
model.nb_data = size(X,1);
%if size(X,1)<size(X,2), warning('less datapoints than dimension of a datapoint ?'); end
%if size(Y,1)<size(Y,2), warning('less targetpoints than dimension of a targetpoint ?'); end
if isempty(Y), error('empty datapoint vector...'); end

%
% initializing kernel type
%
% try中代码错误之后，会执行catch中的代码，catch代码执行完后，会继续执行后续代码，一般用try-catch-end来检查错误或者避免程序中断
try model.kernel_type = kernel_type; catch, model.kernel_type = 'RBF_kernel'; end

%
% using preprocessing {'preprocess','original'}
%
try model.preprocess=preprocess; catch, model.preprocess='preprocess';end
if model.preprocess(1) == 'p', 
  model.prestatus='changed';
else
    model.prestatus='ok'; 
end

%
% initiate datapoint selector
%
model.xtrain = X;
model.ytrain = Y;
model.selector=1:model.nb_data;

%
% regularisation term and kenel parameters
%
if(gam<=0), error('gam must be larger then 0');end
model.gam = gam;

if sig2<=0,
  model.kernel_pars = (model.x_dim);
else
  model.kernel_pars = sig2;
end

%
% dynamic models
%
model.x_delays = 0;
model.y_delays = 0;
model.steps = 1;

% for classification: one is interested in the latent variables or
% in the class labels
model.latent = 'no';

% coding type used for classification
model.code = 'original';
try model.codetype=codetype; catch, model.codetype ='none';end

% preprocessing step
model = prelssvm(model);

% status of the model: 'changed' or 'trained'
model.status = 'changed';

%settings for weight function
model.weights = [];

