function D=Recombination(T,phyx,phyy)
D=zeros(129,129);

for i=2:128
    for j=2:128
        deltx=(phyx(i,j)-fix(phyx(i,j)));
        delty=(phyy(i,j)-fix(phyy(i,j)));
        m1=fix(phyx(i,j));
        m2=fix(phyy(i,j));
        if m1<=1
            m1=1;
        elseif m1>=128
            m1=128;
        end
        if m2<=1
            m2=1;
        elseif m2>=128
            m2=128;
        end
        D(i,j)=(1-deltx)*(1-delty)*double(T(m1,m2))+deltx*(1-delty)*double(T(m1+1,m2))+delty*(1-deltx)*double(T(m1,m2+1))+deltx*delty*double(T(m1+1,m2+1));
    end
end
end
