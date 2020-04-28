function  [ept,Ep,Hp] = cost(Z,P,Target,objfun,defo)
% This function  computes the cost.
%
% Inputs :
%   X.center: is a (n x d) matrix containing the points.
%   X.vector: a cell contains first  and second vectors of the 2-vector.
%   mom.center: is a (n x d) matrix containing the spatial momentums.
%   mom.vector: a cell contains momentums about first  and second vectors of the 2-vector.
%   defo: is a structure of deformations.


% Target: similar to X
% K_q: Positive definite matrix used in computing the Hamiltonian energy.
% Cost can be computed without inputuing K_q

%Output:
%ept.X,ept.P: evolution of states and momemtum variables
%ept.ham: Hamiltonian energy
%ept.dat: data attachment term of the cost function 

 
%ept.cost: Total cost
%Ep: gradient
 [n,~]=size(Z); 
 
 if isfield(defo,'Css') == 0    
  defo.Css = ((repmat(Z(:,1),1,n)-repmat(Z(:,1)',n,1)).^2)/(defo.kernel_size_mom(1,1)^2);
end
 x = Z(:,2);
   [ept.x,ept.mom]=forward_tan(Z,P,defo,1);
   [ept.ham,Hp] = Ham(x,P,defo);
%     Z_end = [Z(:,1),ept.x{end}];
%     ept.dat = norm(Z_end-Target,'fro')^2;
    ept.dat = norm(ept.x{end}-Target(:,2),'fro')^2;
    ept.cost = ept.ham + objfun.lambda*ept.dat;
    
    if nargout >1
       dG_1 =  2*(ept.x{end}-Target(:,2));
       dpfinal = zeros(n,1);
       [~,dpinit]=backward_tan(dG_1,dpfinal,ept,defo);
       Ep = Hp + objfun.lambda*dpinit;
%         Ep(:,1) = zeros(n,1);
    end

end