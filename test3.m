clear; close all;
addpath('C:\Users\NTU_Math\Documents\prelim\color noise code\QuEST_v027\QuEST_v027')
x_0 = 7:-1:5;
r = length(x_0);
k = 4*r;
T = 20;
ps = 100:100:600;
gamma = 0.5;
mF = [];mO = [];mN = [];mS = [];
sF = [];sO = [];sN = [];sS = [];
mS0 = [];mSw = [];mSi = [];
sS0 = [];sSw = [];sSi = [];
mwF = [];mwO = [];mwN = [];
mwF2 = [];
swF = [];swO = [];swN = [];
mA1 = [];mA2 = [];mD = [];
%rng(100)
%name = 'Marcenko-Pastur';
%name = 'Mix2';
%name = 'Unif[1,10]';
%name = 'PaddedIdentity';
name = 'Chi10';
%name = 'Fisher3n';
%name = 'Triangle';

for i = 1:length(ps)
    p = ps(i);
    x = [x_0, zeros(1,p - length(x_0))];
    errF = [];errO = [];errN = [];errSigma = [];
    err_S0 = [];err_Sw = [];err_Si = [];
    err_wF = []; err_wO = []; err_wN = [];
    err_wF2 = [];
    errA1 = []; errA2 = [];errD = [];
    
    for t  = 1:T
        
        n = ceil(p/gamma);
        w = n^(-1/3);
        G = randn(n,n);
        [A, ~, ~] = svd(G);
        A = A(:,1:p);
        G = randn(p,p);
        [B, ~, ~] = svd(G);
        %B = B(:,1:r);
        X = B * diag(x) * A';
        %Sigma = ones(1,p);%sort(abs(randn(1,p)),'descend');
        [Q,~] = svd(randn(p,n));
        %Q = diag(ones(1,p));
        [Z,fT] = Noise(p,n,name,Q);
        Sigma = Q*fT.^2*Q';
        
        [~,sigma] = svd(Sigma);
        sigma = diag(sigma);
        sigma = sort(sigma,'descend');
        Y = X + Z;
        [U,hx,V] = svd(Y);
   
        %[ZZ,~] = Noise(p,n^2,name);
        
     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        eta_f = zeros(1,p);
        eta_o = zeros(1,p);
        eta_n = zeros(1,p);
        u = eig(Y'*Y); u = sort(u,'descend');
        lab = eig(Y*Y'); lab = sort(lab,'descend');
        lambda = eig(Y*Y'); lambda = sort(lambda,'descend');
        
        xx = lambda(1:50)./lambda(2:51)-1;
        r_p = r;%max(find(xx>=w));
        
        %r_p = r;
        [lY, fY, rY] = svd(Y);
        fY = diag(fY);
        fZ = createPseudoNoise(fY, k, 'i');
        hX_si = lY*diag(fZ)*rY(:,1:length(fZ))';
        [~,~,tauhat,~,~,~,~,~,exitflag,numiter,x0]=QuESTimate(sqrt(n)*(lY*diag(fZ)*rY(:,1:length(fZ))')',0);
        sigmahat = Q*diag(tauhat)*Q';
        %hSigma = sort(tauhat,'descend');
        
        hSigma = sigma;
        %hSigma = (ones(1,length(sigma))*max(svd(Z))/(1+sqrt(gamma))).^2;
        wt_sigma_col = [];
        a1_col = []; a2_col = [];
        d_col = [];
        for j = 1:r_p
            theta = u(j);
            wt_sigma = -(1/n *sum(1./(u(r_p+1:end)-u(j))))^(-1);
            %wt_sigma = -(1/n *sum(1./(z-u(j))))^(-1);
            dtheta = -(1/p *sum(1./(u(r_p+1:end)-u(j)).^2))^(-1);
            %m1 = (1/p *sum(1./(lab(r_p+1:end)-u(j))));
            %dm1 = (1/p *sum(1./(lab(r_p+1:end)-u(j)).^2));
            %m2 = (1/n *sum(1./(u(r_p+1:end)-u(j))));
            %dm2 = (1/n *sum(1./(u(r_p+1:end)-u(j)).^2));
            %Tau = u(j)*m1*m2; dTau = m1*m2 + u(j)*dm1*m2 + u(j)*m1*dm2;
            hd = sqrt(wt_sigma - hSigma(j));
            %d = 1/sqrt(u(j)*m1*m2);
            a1 = dtheta/(wt_sigma*theta);
            a2 = hd^2*dtheta/wt_sigma^3;
            d = hd/sqrt(a1*a2);
            %a1 = m2/(d^2*dTau); a2 = m1/(d^2*dTau);
            eta_f(j) = d*sqrt(a1*a2);
            eta_o(j) = d;
            eta_n(j) = abs(d*(sqrt(a1*a2)- sqrt((1-a1)*(1-a2))));
            wt_sigma_col = [wt_sigma_col, wt_sigma];
            err_a1 = abs(abs(U(:,j)'*B(:,j)) - sqrt(abs(a1)));
            err_a2 = abs(abs((V(:,j)'*A(:,j))) - sqrt(abs(a2)));
            a1_col = [a1_col,err_a1];
            a2_col = [a2_col,err_a2];
            d_col = [d_col,d-x(j)];
        end
        
        
        %err_sigma = abs(wt_sigma_col(1:r_p) - sigma(1:r_p)' - x(1:r_p).^2);
        err_sigma = abs(wt_sigma_col(1:r_p) - sigma(1:r_p)' - (x(1:r_p).*diag(U(:,1:r_p)'*B(:,1:r_p)).*diag(V(:,1:r_p)'*A(:,1:r_p))).^2);
        hX_f = U*diag(eta_f)*V(:,1:p)';
        err_f = norm(X-hX_f,'fro');
        hX_o = U*diag(eta_o)*V(:,1:p)';
        err_o = max(svd(X-hX_o));
        hX_n = U*diag(eta_n)*V(:,1:p)';
        err_n = sum(svd(X-hX_n));
        errF = [errF; err_f];
        errO = [errO; err_o];
        errN = [errN; err_n];
        errSigma = [errSigma; err_sigma(1:r_p)];
        errA1 = [errA1;a1_col(1:r)];
        errA2 = [errA2;a2_col(1:r)];
        errD = [errD;d_col(1:r)];
        %ScreeNot
        [hX_s0, ~, ~] = adaptiveHardThresholding(Y, k, '0');
        err_s0 = norm(X-hX_s0,'fro');
        [hX_sw, ~, ~] = adaptiveHardThresholding(Y, k, 'w');
        err_sw = norm(X-hX_sw,'fro');
        [hX_si, ~, ~] = adaptiveHardThresholding(Y, k, 'i');
        err_si = norm(X-hX_si,'fro');
        err_S0 = [err_S0; err_s0];
        err_Sw = [err_Sw; err_sw];
        err_Si = [err_Si; err_si];
        
        %Whitening
        [sU,sD] = svd(Sigma);
        %[sU,sD] = svd(sigmahat);
        invSigma2 = sU*(pinv(sD).^(1/2))*sU';
        Sigma2 = sU*(sD.^(1/2))*sU';
        wY = invSigma2*Y;
        [wU,wD,wV] = svd(wY);
        hD = optimal_shrinkage(diag(wD),gamma,'fro',1);
        whX_fro = wU*diag(hD)*wV(:,1:p)'; hX_fro = Sigma2*whX_fro; 
        err_fro = norm(X-hX_fro,'fro');
        
        
        
        hD = optimal_shrinkage(diag(wD),gamma,'op',1);
        whX_op = wU*diag(hD)*wV(:,1:p)'; hX_op = Sigma2*whX_op; 
        err_op = max(svd(X-hX_op));
        
        hD = optimal_shrinkage(diag(wD),gamma,'nuc',1);
        whX_nuc = wU*diag(hD)*wV(:,1:p)'; hX_nuc = Sigma2*whX_nuc; 
        err_nuc = sum(svd(X-hX_nuc));
        
        
        [sU,sD] = svd(sigmahat);
        invSigma2 = sU*(pinv(sD).^(1/2))*sU';
        Sigma2 = sU*(sD.^(1/2))*sU';
        wY = invSigma2*Y;
        [wU,wD,wV] = svd(wY);
        hD = optimal_shrinkage(diag(wD),gamma,'fro',1);
        whX_fro = wU*diag(hD)*wV(:,1:p)'; hX_fro2 = Sigma2*whX_fro; 
        err_fro2 = norm(X-hX_fro2,'fro');
        
        err_wF = [err_wF; err_fro];
        err_wO = [err_wO; err_op];
        err_wN = [err_wN; err_nuc];
        err_wF2 = [err_wF2; err_fro2];
    end
    mF = [mF, errF];mO = [mO, errO];mN = [mN, errN];
    sF = [sF, std(errF)];sO = [sO, std(errO)];sN = [sN, std(errN)];
    mS0 = [mS0, err_S0];mSw = [mSw, err_Sw];mSi = [mSi,err_Si];
    sS0 = [sS0,std(err_S0)];sSw = [sSw,std(err_Sw)];sSi = [sSi,std(err_Si)];
    mwF = [mwF,err_wF];mwO = [mwO, err_wO];mwN = [mwN,err_wN];
    mwF2 = [mwF2,err_wF2];
    swF = [swF, std(err_wF)];swO = [swO, std(err_wO)];swN = [swN, std(err_wN)];
    
    mD = [mD;median(errD)];
    
    mA1 = [mA1; median(errA1)];
    mA2 = [mA2; median(errA2)];
    
    mS = [mS;median(errSigma)];
    sS = [sS, std(errSigma)];
end
%boxplot(ps, mF)

%semilogy(ps, median(mO),'Linewidth', 5);
%plot(ps, log(median(mO)),'Linewidth', 5);
%{
plot(ps,abs(mS(:,1)));
hold on;
%plot(ps,median(mSi),'Linewidth', 5);
%plot(ps,median(mwF),'Linewidth', 5);
%legend('Our algorithm','ScreeNot', 'Whitening')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Error')
%}
plot(ps, median(mF),'Linewidth', 5);
hold on;
plot(ps,median(mSi),'Linewidth', 5);
plot(ps,median(mwF),'Linewidth', 5);
plot(ps,median(mwF2),'Linewidth', 5);
legend('Our algorithm','ScreeNot', 'Oracle Whitening', 'Estimate Whitening')
set(gca,'FontSize',12)
xlabel('n');
ylabel('Error')
%{
plot(ps,mS(:,1),'Linewidth',7)
hold on
plot(ps,mS(:,2),'Linewidth',7)
plot(ps,mS(:,3),'Linewidth',7)
plot(ps,mS(:,4),'Linewidth',7)
plot(ps,mS(:,5),'Linewidth',7)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlabel('n')
ylabel('Error')

mA1 = abs(mA1);
plot(ps,mA1(:,1),'Linewidth',2)
hold on
plot(ps,mA1(:,2),'Linewidth',2)
plot(ps,mA1(:,3),'Linewidth',2)
plot(ps,mA1(:,4),'Linewidth',2)
plot(ps,mA1(:,5),'Linewidth',2)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlabel('n')
ylabel('Error')
axis tight
title('Left Cosine Error')
%}
%{
mA2 = abs(mA2);
plot(ps,mA2(:,1),'Linewidth',2)
hold on
plot(ps,mA2(:,2),'Linewidth',2)
plot(ps,mA2(:,3),'Linewidth',2)
plot(ps,mA2(:,4),'Linewidth',2)
plot(ps,mA2(:,5),'Linewidth',2)
legend('error 1', 'error 2','error 3','error 4','error 5')
set(gca,'FontSize',20)
xlabel('n')
ylabel('Error')
axis tight
title('Right Cosine Error')

%}