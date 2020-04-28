function P_final = get_momentum(src, tar, xkernel, ykernel)
    defo = struct();
    options = struct();
    defo.kernel_size_mom   = [xkernel ykernel];
    defo.nb_euler_steps    = 15;

    objfun.lambda = 1.5;

    options.record_history = true;
    options.maxit          = 100;
    options.nvec           = 10;
    options.prtlevel       = 0;

    P_init = -rand(size(src,1),1); 
    [P_final,~] = vmatch(src,tar,P_init,defo,objfun,options);
    P = P_final(ceil(size(src,1)/2));
end