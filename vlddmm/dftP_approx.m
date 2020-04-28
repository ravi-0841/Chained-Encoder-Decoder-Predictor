function [dPx,dPp]=dftP_approx(x,p,Px,Pp,defo)
% This computes the system [dPx;dPp] = dft([x;p]) * [Px;Pp] via finite
% difference (See eg. [Arguillere, Trelat, Trouve, Younes. Shape deformation 
% analysis from the optimal control viewpoint] Proposition 9 at http://arxiv.org/abs/1401.0661)
%
% Inputs :
%   x: is a (n x d) matrix containing the points.
%   p: is a (n x d) matrix containing the momentums.
%   Px : is a (n x d) matrix (adjoint variable x).
%   Pp : is a (n x d) matrix  (adjoint variable  momentums ).
%   defo : structure containing the deformations parameters
%
% Outputs
%   dPx : (n x d) matrix containing the update of Px.
%   dPp :(n x d) matrix containing the update of Pp.

% finite diff perturbation, use double to increase precision
hh = 1e-7;
defo.precision = 'double';

diff=@(ph,mh,hh) (ph-mh) / (2*hh);

% for lam = defo.kernel_size_mom
    
    
   [dxxmhP,dxpmhP] =  dHr_tan(x-hh*Pp,p+hh*Px,defo);
   [dxxphP,dxpphP] =  dHr_tan(x+hh*Pp,p-hh*Px,defo);

   dPx = diff(dxxphP,dxxmhP,hh);
   dPp = diff(dxpphP,dxpmhP,hh);
   
% end

end