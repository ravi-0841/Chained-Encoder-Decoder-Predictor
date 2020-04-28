function [f,g] = obj_cost(p)
% Simple wrapper function for cost that can be called by Hanso bfgs.
  global Z_1 Y_1 objfun_1 defo_1 
      
%     [n,d] = size(Z_1);
   

    if nargout > 1 % gradient required
        [ept,g] = cost(Z_1,p,Y_1,objfun_1,defo_1);        
    else
        ept = cost(Z_1,p,Y_1,objfun_1,defo_1);

    end
    
     f = ept.cost;
end