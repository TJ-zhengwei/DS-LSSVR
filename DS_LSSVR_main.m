clc
clear
close all
%==================================================================================
type = 'function estimation';
rand('state', sum(100 * clock));
%-------------------------------------------------------------%
% training data
Ntrain=3000;
x1=rand(Ntrain,1);
x2=rand(Ntrain,1);
x3=rand(Ntrain,1);
x4=rand(Ntrain,1);
X=[x1 x2 x3 x4];
Y=exp((x1.^2+x2.^2)*2)+sin(x3)+2*x4;
%-------------------------------------------------------------%
% testing data
Ntest=100;
xtest1=rand(Ntest,1);
xtest2=rand(Ntest,1);
xtest3=rand(Ntest,1);
xtest4=rand(Ntest,1);
Xtest=[xtest1 xtest2 xtest3 xtest4];
Ytest=exp((xtest1.^2+xtest2.^2)*2)+sin(xtest3)+2*xtest4;
%==================================================================================
% first stage of sparsity
rownum = size(X,1);
nc = 4;
ev = grey_relation_data(X);
% Arrange in descending order according to grey relational entropy
[evsort_d, orinum_d] = sort(ev,'descend');  
srd = round(rownum * 0.8);  % Taking the first 80% of the data, the sparse rate is equivalent to 20%
Xtr = zeros(srd,nc);
Ytr = zeros(srd,1);
for i = 1:srd
    Xtr(i,:) = X(orinum_d(i),:);
    Ytr(i) = Y(orinum_d(i));
end
sra = rownum - srd;  % Take the last 20% of the data
% 取出后20%的数据的1%作为底层样本
bnum = round(sra*0.01);
bksra = sra - bnum;
bkevk = zeros(bksra,1);
for i=1:bksra;
    bkevk(i)=evsort_d(srd+i);
end
XtrKB = Xtr;
YtrKB = Ytr;
j = 1;
for i = 1:bnum
    XtrKB(srd+j,:) = X(orinum_d(srd+bksra+i),:);
    YtrKB(srd+j) = Y(orinum_d(srd+bksra+i));
    j = j + 1;
end
sr = srd + bnum;
% initial value of K is set to 5 according to the experience of tests 
K = 5;
for ite = 1:100
    [idx,cp,sumd,dis]=kmeans(bkevk,K);
    kn = 1;
    for k = 1:K
        num = sumd(k);
        j = idx(kn);
        sumdis = 0;
        for i = 1:bksra 
            if j == idx(i)
                kn = kn + 1;
            end
            if j ~= idx(i)
                sumdis = sumdis + dis(i,j);
            end
        end
        den = sumdis;
        Gb(k) = num / den;
    end
    GKB(ite) = sum(Gb);
    if ite > 1
        if GKB(ite-1)<GKB(ite)
            bestKB = K;
            break;
        end
    end
    K = K + 1;
end
% call the function of kmeans
[idxb,cpb,sumdb,disb]=kmeans(bkevk,bestKB);
j = 0;
k = 1;
for i =1:bksra
    if j ~= idxb(i)
       evsort_indexb = find(min(disb(:,idxb(i)))==disb(:,idxb(i)));
       data_indexb = orinum_d(evsort_indexb);
       XtrKB(sr+k,:) = X(data_indexb,:);
       YtrKB(sr+k) = Y(data_indexb);
       j = idxb(i);
       k = k +1;
    end
end
sr = sr + bestKB;
%-------------------------------------------------------------%
% second stage of sparsity
sp05 = 0.5;
ws = 0.5;
[bestvalue0505,sprate0505,bestp0505,bestalpha0505,bestb0505] = pso_sparse(XtrKB,YtrKB,sp05,ws);
j = 1;
for i = 1:sr
    if bestp0505(i) >= sp05
        xs0505(j,:)=XtrKB(i,:);
        ys0505(j,:)=YtrKB(i,:);
        j = j + 1;
    end
end
tic;
Yppso_lssvr0505 = simlssvm({xs0505,ys0505,type,bestp0505(sr+1),bestp0505(sr+2),'RBF_kernel','preprocess'},{bestalpha0505,bestb0505},Xtest);
time_yppso0505 = toc;
yppsoerror0505 = sqrt(mse(Yppso_lssvr0505 - Ytest));
%==================================================================================
% deep sparsity is carried out directly without shallow sparsity
sp05 = 0.5;
ws = 0.5;
[bestvalue,sprate,bestp,bestalpha,bestb] = pso_sparse(X,Y,sp05,ws);
j = 1;
for i = 1:Ntrain
    if bestp(i) >= sp05
        xs(j,:)=X(i,:);
        ys(j,:)=Y(i,:);
        j = j + 1;
    end
end
tic;
Yppso_lssvr = simlssvm({xs,ys,type,bestp(Ntrain+1),bestp(Ntrain+2),'RBF_kernel','preprocess'},{bestalpha,bestb},Xtest);
time_yppso = toc;
yppsoerror = sqrt(mse(Yppso_lssvr - Ytest));