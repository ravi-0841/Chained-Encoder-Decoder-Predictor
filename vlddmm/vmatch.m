function[P,summary] = vmatch(Z,Y,P_ini,defo,objfun,options)
 global Z_1 Y_1 objfun_1 defo_1
 Z_1 = Z; Y_1 = Y; 
 objfun_1 = objfun; 
 defo_1 = defo;
 
%  [n,d] = size(Z);
 
  if nargin<6
   options.record_history =true;
   options.nvec = 10;
   options.prtlevel = 2;
  end
  
  defo_1.Kq = Kqop(Z_1,defo_1);
   
%   p_0 = stru2vec(P_ini);
%   p_0 = P_ini;
  [P,summary] = perform_bfgs('obj_cost', P_ini, options);
%   P = reshape(p(1:n*d),d,n)';
%   P = vec2stru(p,n,d);

end