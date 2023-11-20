
function [ev] = grey_relation_data(alldata)

[m,n] = size(alldata);
X_de=zeros(m,n); 
X_co=zeros(m,n); 
% 1
for i = 1:m
    X_rmax(i) = max(alldata(i,:));
end
% 2
i=1;
while(i ~= m + 1)
    for j=1:1:n
        X_de(i,j)=abs(alldata(i,j)-X_rmax(i));
    end
    i=i+1;
end
% 3
error_min=min(min(X_de));
error_max=max(max(X_de));
% 4
i=1;
p=0.5;
while(i~=m+1)
    for j=1:1:n
        X_co(i,j)=(error_min+p*error_max)/(X_de(i,j)+p*error_max);
    end
    i=i+1;
end
% 5
a=zeros(1,m);
p=zeros(m,n);
for i=1:1:m  
    for j=1:1:n
        a(i)=a(i)+(1/X_co(i,j));     
    end;
    for j=1:1:n
        p(i,j)=(1/X_co(i,j))/(1/a(i));
    end
end
ev=zeros(1,m);
for i = 1:m
    for j = 1:n;
        ev(i)=ev(i) + p(i,j)*log(p(i,j));
    end
    ev(i)=(1/log(n))*ev(i);
end
[c,s] = sort(ev);
% figure
figure(1);
plot(ev);
figure(2);
bar(ev); 