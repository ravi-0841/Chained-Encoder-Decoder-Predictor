function K_q = Kqop(Z,defo)

    [n,d] = size(Z);
    sig = defo.kernel_size_mom;
    S=zeros(n);
    
     for l=1:d
        S=S+((repmat(Z(:,l),1,n)-repmat(Z(:,l)',n,1)).^2)/(sig(1,l)^2);
     end

    K_q =  exp(-S);
end