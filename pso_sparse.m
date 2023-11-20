
% contributive and weighted PSO algorithm
function [minfit,maxsp,output,outalpha,outb] = pso_sparse(XModel,YModel,sp,ws) 
c1 = 1.8; 
c2 = 1.8;
w = 0.9;  
MaxDT = 500; 
[nr nc] = size(XModel);
D=nr+2; 
xmax=1*ones(1,D);
xmin=0*ones(1,D);
xmax(D-1) = 10000000000;
xmin(D-1) = 1;
xmax(D) = 100;
xmin(D) = 0.1;
N = 100; 
eps = 0.0000001; 
vmax = xmax/2;
vmin = -vmax;
miny = Inf;
alpha = cell(N,1);  
%------
for i = 1:N
  for j = 1:D-2
      x(i,j) = rand;
      v(i,j) = rand;
  end
  j = D - 1;
  x(i,j) = 1 + 10000000000 * rand; 
  v(i,j) = 10000000000 * rand;        
  j = D;
  x(i,j) = 0.1 + 100 * rand; 
  v(i,j) = 100 * rand;        
end
%------
for i = 1 : N
    k = 1;
    Xs = [];
    Ys = [];
    for j = 1:(D-2)
        if x(i,j) >= sp
          Xs(k,:) = XModel(j,:);
          Ys(k,:) = YModel(j,:);
          k = k + 1;
        end
    end
    if k == 1
      break;
    end
    [y1(i),alpha{i},b(i)] = lssvrfit_pso_sp(Xs, Ys, x(i,D-1), x(i,D), XModel, YModel); 
    pb(i,:) = x(i,:); 
    y2(i) = (k - 1) / nr;
    y(i) = (1-ws)*y1(i) + ws*y2(i);
    if y(i) < miny
        miny = y(i);
        pg = x(i,:);
        alphabest = alpha{i};
        bbest = b(i);
        spr = 1 - y2(i);
    end
end
%------
for t = 1 : MaxDT
  w = 0.9 - 0.5 * t / MaxDT; 
  for i = 1 : N
    v(i,:) = w * v(i,:) + c1 * rand * (pb(i,:) - x(i,:)) + c2 * rand * (pg - x(i,:));
    for j = 1 : D
        if v(i,j) > vmax(j)
            v(i,j) = vmax(j);
        elseif v(i,j) < vmin(j)
            v(i,j) = vmin(j);
        end
    end
    x(i,:) = x(i,:) + v(i,:);
    for j = 1 : D
        if x(i,j) > xmax(j)
            x(i,j) = xmax(j);
        elseif x(i,j) < xmin(j)
            x(i,j) = xmin(j);
        end
    end
    k = 1;
    Xs = [];
    Ys = [];
    for j = 1:(D-2)
        if x(i,j) >= sp
          Xs(k,:) = XModel(j,:);
          Ys(k,:) = YModel(j,:);
          k = k + 1;
        end
    end
    if k == 1
      break;
    end
    [yt1,alphat,bt] = lssvrfit_pso_sp(Xs,Ys,x(i,D-1),x(i,D),XModel, YModel);
    yt2 = (k - 1) / nr;
    yt = (1-ws)*yt1 + ws*yt2;
    if yt < y(i)
      y(i) = yt;
      pb(i,:) = x(i,:);
      alpha{i} = alphat;
      b(i) = bt;
      y2(i) = yt2;
    end
    if y(i) < miny
        miny = y(i);
        pg = pb(i,:);
        alphabest = alpha{i};
        bbest = b(i);
        spr = 1 - y2(i);
    end
  end

  if miny < eps
      output = pg;
      outalpha = alphabest;
      outb = bbest;
      minfit = miny;
      maxsp = spr;
      return;
  end
end
output = pg;
outalpha = alphabest;
outb = bbest;
minfit = miny;
maxsp = spr;