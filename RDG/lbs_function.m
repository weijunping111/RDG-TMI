function [phyx,phyy,ssd_mu]=lbs_function(rou,tau,alpha)
N1=129;
N2=129;
N=129;
% lamda=0.5;
theta=1;
dt=0.001;
% phyx=Meshx;
% phyy=Meshy;
% [D2u,D1u]=gradient(Meshx)
% [D2v,D1v]=gradient(Meshy)
% rou=(D1u.^2-D2u.^2+D1v.^2-D2v.^2)./((D1u+D2v).^2+(D2u-D1v).^2);%;
% 
% tau=2*(D1u.*D2u+D1v.*D2v)./((D1u+D2v).^2+(D2u-D1v).^2);%;
s=1000000;
m=0.1;  %异常值阈值
ssd_f=129*129;
phyy=zeros(N1,N2);
phyx=zeros(N1,N2);
h=zeros(N1,N2);
f=zeros(N1,N2);
u=zeros(129,129);
v=zeros(129,129);
D1u=zeros(129,129);
D1v=zeros(129,129);
D2u=zeros(129,129);
D2v=zeros(129,129);
D11u=zeros(129,129);
D11v=zeros(129,129);
D22u=zeros(129,129);
D22v=zeros(129,129);
D12u=zeros(129,129);
D12v=zeros(129,129);
uuu=zeros(129,129);
vvv=zeros(129,129);
uu=zeros(129,129);
vv=zeros(129,129);

for i=1:N
    for j=1:N

        phyy(i,j)=j;
        phyx(i,j)=i;
    end
end
%
% u=rand(129,129)+1;
% v=rand(129,129)+1;
% u=zeros(129,129);
% v=zeros(129,129);

filter2=fspecial('gaussian',3,100);

filter1=fspecial('average');

%随机生成满足条件的beltrami系数
% mu1= abs(2*rand(N1+1, N2+1)-1);
% mu2= abs(2*rand(N1+1, N2+1)-1);
% mu1=imfilter(mu1,filter1);
% mu2=imfilter(mu2,filter1);
% mu1=imfilter(mu1,filter1);
% mu2=imfilter(mu2,filter1);
% squared_sum = mu1.^2 +mu2.^2;
% max_squared_sum = max(squared_sum, [], 'all');
% % if max_squared_sum> 1
% rou=mu1./(max_squared_sum+0.1);
% tau=mu2/(max_squared_sum+0.1);
% load('tau_lam=10.mat')
% load('rou_lam=10.mat')


% for i=1:N+1
%     rou(i,1:3)=0;
%     rou(i,N-2:N+1)=0;
%     rou(1:3,i)=0;
%     rou(N-2:N+1,i)=0;
%     tau(i,1:3)=0;
%     tau(i,N-2:N+1)=0;
%     tau(1:3,i)=0;
%     tau(N-2:N+1,i)=0;
% end

for kk=1:1
    lamda=100; %0.4*0.8^(kk)*(1.1^(kk-1))

    kk

    for k=1:30000

        %     [D1u,D2u]=gradient(u);
        %     [D11u,D12u]=gradient(D1u);
        %     [~,D22u]=gradient(D2u);
        %     [D1v,D2v]=gradient(v);
        %     [D11v,D12v]=gradient(D1v);
        %     [~,D22v]=gradient(D2v);

        %
        % for i=2:N-1
        % for j=2:N-1
        %
        % D1u(i,j)=(phyx(i+1,j)-phyx(i-1,j))/2;
        % D1v(i,j)=(phyy(i+1,j)-phyy(i-1,j))/2;
        %
        % D2u(i,j)=(phyx(i,j+1)-phyx(i,j-1))/2;
        % D2v(i,j)=(phyy(i,j+1)-phyy(i,j-1))/2;
        %
        % D11u(i,j)=(phyx(i+1,j)-2*phyx(i,j)+phyx(i-1,j));
        % D11v(i,j)=(phyy(i+1,j)-2*phyy(i,j)+phyy(i-1,j));
        % D12u(i,j)=(phyx(i+1,j+1)-phyx(i-1,j+1)-phyx(i+1,j-1)+phyx(i-1,j-1))/4;
        % D12v(i,j)=(phyy(i+1,j+1)-phyy(i-1,j+1)-phyy(i+1,j-1)+phyy(i-1,j-1))/4;
        % D22u(i,j)=(phyx(i,j+1)-2*phyx(i,j)+phyx(i,j-1));
        % D22v(i,j)=(phyy(i,j+1)-2*phyy(i,j)+phyy(i,j-1));
        % end
        % end

        [D2u,D1u]=gradient(phyx);
        [D12u,D11u]=gradient(D1u);
        [D22u,~]=gradient(D2u);
        [D2v,D1v]=gradient(phyy);
        [D12v,D11v]=gradient(D1v);
        [D22v,~]=gradient(D2v);

        % for i=1:N1
        %     for j=1:N2
        %     h(i,j)=-2*(rou(i,j)*((D1u(i,j)+D2v(i,j))^2+(D2u(i,j)-D1v(i,j))^2)-(D1u(i,j)^2-D2u(i,j)^2+D1v(i,j)^2-D2v(i,j)^2))*(2*(rou(i,j)-1)*D11v(i,j)+2*(rou(i,j)+1)*D22v(i,j))-2*(tau(i,j)*((D1u(i,j)+D2v(i,j))^2+(D2u(i,j)-D1v(i,j))^2)-(2*D1u(i,j)*D2u(i,j)+2*D1v(i,j)*D2v(i,j)))*(2*tau(i,j)*D11v(i,j)+2*tau(i,j)*D22v(i,j)-4*D12v(i,j));
        %     f(i,j)=-2*(rou(i,j)*((D1u(i,j)+D2v(i,j))^2+(D2u(i,j)-D1v(i,j))^2)-(D1u(i,j)^2-D2u(i,j)^2+D1v(i,j)^2-D2v(i,j)^2))*(2*(rou(i,j)-1)*D11u(i,j)+2*(rou(i,j)+1)*D22u(i,j))-2*(tau(i,j)*((D1u(i,j)+D2v(i,j))^2+(D2u(i,j)-D1v(i,j))^2)-(2*D1u(i,j)*D2u(i,j)+2*D1v(i,j)*D2v(i,j)))*(2*tau(i,j)*D11u(i,j)+2*tau(i,j)*D22u(i,j)-4*D12u(i,j));
        % %%%%异常值处理

        f=-2*(((rou-1).^2+tau.^2).*D11u-4*tau.*D12u+((rou+1).^2+tau.^2).*D22u);
        h=-2*(((rou-1).^2+tau.^2).*D11v-4*tau.*D12v+((rou+1).^2+tau.^2).*D22v);
        [by,bx]=gradient((rou-1).^2+tau.^2);
        [cy,cx]=gradient((rou+1).^2+tau.^2);
        [dy,dx]=gradient(rou.^2+tau.^2);
        [ty,tx]=gradient(tau);

        f=f-2*(D1u.*(bx-2*ty)+D2u.*(cy-2*tx)-D1v.*dy+D2v.*dx);
        h=h-2*(D1u.*dy-D2u.*dx-D1v.*(bx-2*ty)+D2v.*(cy-2*tx));



        % l=(((D1u+D2v).^2+(D2u-D1v).^2).*rou-(D1u.^2-D2u.^2+D1v.^2-D2v.^2)).^2+(((D1u+D2v).^2+(D2u-D1v).^2).*tau-2*(D1u.*D2u+D1v.*D2v)).^2;
        % lamdal=sumsqr(l)/129/129;

        % lamdaf=sumsqr(f)/129/129;
        % lamdah=sumsqr(h)/129/129;

        f=lamda*f;
        h=lamda*h;
        % f=rou-u1;
        % h=tau-u2;
        % lamda=sumsqr(f)/129/129;

        % for i=1:N1
        %     for j=1:N2
        % if h(i,j)>m
        %     h(i,j)=m;
        % end
        % if h(i,j)<-m
        %     h(i,j)=-m;
        % end
        % if f(i,j)>m
        %     f(i,j)=m;
        % end
        % if f(i,j)<-m
        %     f(i,j)=-m;
        % end
        %     end
        % end
        ux=MCG1Dx(f,dt,theta,u);
        uxy=MCG1Dy(f,dt,theta,ux);
        vx=MCG1Dx(h,dt,theta,v);
        vxy=MCG1Dy(h,dt,theta,vx);

        u=uxy;
        %     u=imfilter(u,filter1);

        v=vxy;
        %     v=imfilter(v,filter1);

        u(1,:)=0;
        u(:,1)=0;
        u(129,:)=0;
        u(:,129)=0;
        v(1,:)=0;
        v(:,1)=0;
        v(129,:)=0;
        v(:,129)=0;


        for i=1:N
            for j=1:N

                phyy(i,j)=j+v(i,j);
                phyx(i,j)=i+u(i,j);
            end
        end
        % for i=1:N
        % for j=1:N
        %     if phyy(i,j)>N
        %         phyy(i,j)=N;
        %         v(i,j)=N-j;
        %     end
        %     if phyx(i,j)>N
        %         phyx(i,j)=N;
        %         u(i,j)=N-i;
        %     end
        %     if phyy(i,j)<1
        %         phyy(i,j)=1;
        %         v(i,j)=1-j;
        %     end
        %     if phyx(i,j)<1
        %         phyx(i,j)=1;
        %         u(i,j)=1-i;
        %     end
        %     phyy(i,1)=1;
        %     phyy(i,129)=129;
        %     phyx(1,j)=1;
        %     phyx(129,j)=129;
        % end
        % end


        [D2u,D1u]=gradient(phyx);
        [D2v,D1v]=gradient(phyy);

        u1=(D1u.^2-D2u.^2+D1v.^2-D2v.^2)./((D1u+D2v).^2+(D2u-D1v).^2);%;

        u2=2*(D1u.*D2u+D1v.*D2v)./((D1u+D2v).^2+(D2u-D1v).^2);%;


        % u11=uu-u;
        % u22=vv-v;
        % f=-u11;
        % h=-u22;
        k
        ssd_mu=sum(sum((rou-u1).^2+(tau-u2).^2));
        if s-ssd_mu<alpha
            s-ssd_mu;
            fprintf('mu = %f\n', max(max(u1.^2+u2.^2)));
            
            break
        end
        s=ssd_mu;
        fprintf('ssd_mu = %f\n', s);
        % ssd_f=sum(sum((u_true-u).^2+(v_true-v).^2))/129/129
        % sd=sum(sum(abs(u_true-u)+abs(v_true-v)))/129/129
        % maxd=max(max(max(abs(u_true-u))),max(max(abs(v_true-v))))
        % ssd=0;
        % SSD=((rou-1).*D1u+(rou+1).*D2v-tau.*(D1v-D2u)).^2+((rou-1).*D1v-(rou+1).*D2u+tau.*(D1u+D2v)).^2;
        % ssd=sum(sum(SSD))
    end

end




% imagesc(u)
%imagesc(abs(T_i-D))


end

