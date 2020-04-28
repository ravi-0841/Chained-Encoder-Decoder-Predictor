function  [ham,ham_grad] = Ham(Z,P,defo)

   
   [n,d]=size(Z);

   if isfield(defo,'Kq') == 0
      K_q = Kqop(Z,defo); 
   else   
      K_q = defo.Kq;
   end
      
%     p = stru2vec(P);
    ham_grad = K_q*P;
    ham= 0.5*(P'*ham_grad);
%     ham_grad = vec2stru(ham_grad,n,d);

end
