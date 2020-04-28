function [dxHr,dpHr] = dHr_tan(x,P,defo)
% Matlab version of the reduced Hamiltonian system.
% Input : 
%   x : state (n x d) matrix
%   p : costate (n x d) matrix
%   defo : structure containing the field and 'kernel_size_mom' (kernel size)
%
% Output
%   dxHr : gradient of Hr wrt to x at point (x,p)
%   dpHr: gradient of Hr wrt to p at point (x,p)


[n,~]=size(x);

% Calcul de A=exp(-|x_i -x_j|^2/(lam^2))

sig = defo.kernel_size_mom;
% S=zeros(n);
    

S = defo.Css + ((repmat(x,1,n)-repmat(x',n,1)).^2)/(sig(1,2))^2;
A =  exp(-S);
B =  -exp(-S)/sig(1,2)^2;
dpHr=A*P;

Cxx = repmat(x,1,n)-repmat(x',n,1);
dxHr = 2 *sum(B.*Cxx.*(P*P'),2);


end