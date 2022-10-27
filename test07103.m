clear; close all;
cd('C:\Users\NTU_Math\Documents\prelim\color noise code')
 

ps = 6:2:18;
x = [3,3,3,2.1,2.1,2.1, 1.5,1.5,1.5,0.5,0.5,0.5]; 
%gamma = 1; name1 = 'Unif[1,10]';name2 = 'Mix2.2';% r = 5
%gamma = 0.5;name1 = 'Unif[1,10]';name2 = 'Mix2.2';% r = 5
%gamma = 1;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur';
gamma = 0.5;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur';


n = 200; p = floor(n*gamma);
r = length(x);
cm = 10; % student t 10 's moment
c = 1;
smth = '';
rng(1);

method = 'imp';
sh = 'op';

if sh == "fro"
    err_method = 'fro';
else
    err_method = 2;
end


T =50;

mF = [];mO = [];mN = [];mF2 = [];
sF = [];sO = [];sN = [];sF2 = [];
mF0 = [];mO0 = [];mN0 = [];
sF0 = [];sO0 = [];sN0 = [];
mS0 = [];mSw = [];mSi = [];
sS0 = [];sSw = [];sSi = [];
mwF = [];mwO = [];mwN = [];mwF2 = [];
swF = [];swO = [];swN = [];swF2 = [];
mD1 = [];mA1 = [];mD2 = [];mA2 = [];
mD1_2 = [];mA1_2 = [];mD2_2 = [];mA2_2 = [];
mrSN = [];mrp = [];mrold = [];mrW = [];
mDSN1 = [];mASN1 = [];mDSN2 = [];mASN2 = [];
mDW1 = [];mAW1 = [];mDW2 = [];mAW2 = [];mDW1_2 = [];mDW2_2 = [];
mD01 = [];mA01 = [];mD02 = [];mA02 = [];
for i = 1:length(ps)
    kk = ps(i);
    errF = [];errO = [];errN = [];errD = [];errD2 = [];errF2 = [];
    errF0 = [];errO0 = [];errN0 = [];
    err_S0 = [];err_Sw = [];err_Si = [];
    err_wF = []; err_wO = []; err_wN = [];err_wF2 = [];
    errA = []; errA2 = [];err_rSN = [];err_rp = [];
    errDSN = []; errDW = []; errD0 = [];errDW2 = [];
    for t  = 1:T
        
      
       
        G = randn(n,n);
        [V0, ~] =qr(G);
        V0 = V0(:,1:r);
        
        G = randn(p,p);
        [U0, ~] = qr(G);
        U0 = U0(:,1:r);
        
        X = U0 * diag(x) * V0';
        N = trnd(cm,p,n)./(sqrt(1.25)*sqrt(n));
        %student t 10/poisson mean 0/

        [Q1,~] = qr(randn(p,p));% modified QR decomposition
        %Q1 = diag(ones(1,p));%
        [Q2,~] = qr(randn(n,n));
        %Q2 = diag(ones(1,n));%
        [~,fT1] = Noise(p,p,name1,Q1);
        [~,fT2] = Noise(n,n,name2,Q2);
        A = Q1*fT1*Q1';
        B = Q2*fT2*Q2';
        sigma1 = sqrt(sum(diag(fT1).^2)/p);
        sigma2 = sqrt(sum(diag(fT2).^2)/n);
        A = A./sigma1; B = B./sigma2; 
        Y  = X + A*N*B;

        [U,hx,V] = svd(Y);
        hx = diag(hx);
        %[ZZ,~] = Noise(p,n^2,name);
        
     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %[~, ~, r_p] = adaptiveHardThresholding(Y, k, 'i');
        %[~,eta_f] = optimal_shrinkage_color2(Y,sh,r_p,method);
        [~,eta_f,~,~] = optimal_shrinkage_color5(Y,sh,method);
        hX_f = U*diag(eta_f)*V(:,1:p)';
        err_f = norm(X-hX_f,err_method);
        errF = [errF; err_f];

        [~,eta_f2,~,~] = optimal_shrinkage_color5(Y,sh,'cut');
        hX_f2 = U*diag(eta_f2)*V(:,1:p)';
        err_f2 = norm(X-hX_f2,err_method);
        errF2 = [errF2; err_f2];

        %[~,eta_o] = optimal_shrinkage_color2(Y,'op',r_p,method);
        [~,eta_o,r_p,eta_a] = optimal_shrinkage_color4(Y,'op',2*kk,method,cm,c);
        [~,eta_o2,~,eta_a2] = optimal_shrinkage_color4(Y,'op',kk,'cut',cm,c);
        a = [];
        for k = 1:length(x)
            a = [a, (U(:,k)'*U0(:,k)* V(:,k)'*V0(:,k))];
        end
        errD = [errD;abs(eta_o(1:length(x))-x)];
        errA = [errA;abs(eta_a(1:length(x))-a)];
        errD2 = [errD2;abs(eta_o2(1:length(x))-x)];
        errA2 = [errA2;abs(eta_a2(1:length(x))-a)];
        eta_f0 = optimal_shrinkage(hx,gamma,sh);
        hX_f0 = U*diag(eta_f0)*V(:,1:p)';
        err_f0 = norm(X-hX_f0,err_method);
        errF0 = [errF0; err_f0];
        errD0 = [errD0;abs(eta_f0(1:length(x))'-x)];
      

        [hX_si, ~, rSN] = adaptiveHardThresholding(Y, kk, 'i');
        %hy = [hx(1),hx(2),zeros(1,length(hx)-2)];
        %hX_si = U*diag(hy)*V(:,1:p)';
        err_rSN = [err_rSN;rSN];err_rp = [err_rp;r_p];
        err_si = norm(X-hX_si,err_method);
        err_Si = [err_Si; err_si];
        eta_sn = svd(hX_si);
        errDSN = [errDSN;abs(eta_sn(1:length(x))'-x)];
        %Whitening
        hA = diag(sum(Y.^2,2)); hB = diag(sum(Y.^2)/sum(hA(:))*n);
        %invSigma2 = Q1*(pinv(fT1))*Q1';
        %Sigma2 = A;
        wY =  diag(diag(hA).^(-1/2))*Y*diag(diag(hB).^(-1/2));
        [wU,wD,wV] = svd(wY);
        hD = optimal_shrinkage(diag(wD),gamma,sh,1);
        whX_fro = wU*diag(hD)*wV(:,1:p)'; hX_fro = (hA.^(1/2))*whX_fro*(hB.^(1/2)); 
        [~,whD,~] = svd(hX_fro);whD = diag(whD);
        whX_fro2 = optimal_shrinkage_color4(wY,sh,kk,'cut',cm,c);hX_fro2 = (hA.^(1/2))*whX_fro2*(hB.^(1/2));
        [~,whD2,~] = svd(hX_fro2);whD2 = diag(whD2);

        err_fro = norm(X-hX_fro,err_method);
        err_fro2 = norm(X-hX_fro2,err_method);
        err_wF = [err_wF; err_fro];
        err_wF2 = [err_wF2; err_fro2];
        errDW = [errDW;abs(whD(1:length(x))'-x)];
        errDW2 = [errDW2;abs(whD2(1:length(x))'-x)];
        
    end
    mF = [mF, errF];sF = [sF, std(errF)];
    mF2 = [mF2, errF2];sF2 = [sF2, std(errF2)];
    mF0 = [mF0, errF0];sF0 = [sF0, std(errF0)];
    mSi = [mSi,err_Si];sSi = [sSi,std(err_Si)];
    mwF = [mwF,err_wF];swF = [swF, std(err_wF)];
    mwF2 = [mwF2,err_wF2];swF = [swF2, std(err_wF2)];
    mrSN = [mrSN, err_rSN];mrp = [mrp, err_rp];
    
    mD1 = [mD1, errD(:,1)];mD2 = [mD2, errD(:,4)];
    mD1_2 = [mD1_2, errD2(:,1)];mD2_2 = [mD2_2, errD2(:,4)];
    mDSN1 = [mDSN1, errDSN(:,1)];mDSN2 = [mDSN2, errDSN(:,4)];
    mDW1 = [mDW1, errDW(:,1)];mDW2 = [mDW2, errDW(:,4)];
    mDW1_2 = [mDW1_2, errDW2(:,1)];mDW2_2 = [mDW2_2, errDW2(:,4)];
    mD01 = [mD01, errD0(:,1)];mD02 = [mD02, errD0(:,4)];
    mA1 = [mA1, errA(:,1)];mA2 = [mA2, errA(:,4)];
    mA1_2 = [mA1_2, errA2(:,1)];mA2_2 = [mA2_2, errA2(:,4)];
    
end
%boxplot(ps, mF)
errorbar(ps,median(mF),median(mF)-quantile(mF,.25),quantile(mF,.75)-median(mF),'k','Linewidth', 5);
hold on
errorbar(ps,median(mF2),median(mF2)-quantile(mF2,.25),quantile(mF2,.75)-median(mF2),'b','Linewidth', 4.5);
errorbar(ps,median(mSi),median(mSi)-quantile(mSi,.25),quantile(mSi,.75)-median(mSi),'g','Linewidth', 4.5);
errorbar(ps,median(mF0),median(mF0)-quantile(mF0,.25),quantile(mF0,.75)-median(mF0),'m','Linewidth', 4);
errorbar(ps,median(mwF),median(mwF)-quantile(mwF,.25),quantile(mwF,.75)-median(mwF),'c','Linewidth', 3.5);
errorbar(ps,median(mwF2),median(mwF2)-quantile(mwF2,.25),quantile(mwF2,.75)-median(mwF2),'r','Linewidth', 3);
%errorbar(ps, median(mF),median(sF),'k','Linewidth', 5);
%errorbar(ps,median(mF0),median(sF0), 'm','Linewidth', 5);
%errorbar(ps,median(mwF),median(swF),'#D95319','Linewidth', 5);
%legend('Our algorithm','Original Optimal Shrinkage','ScreeNot', 'Whitening','Location','northwest')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Error')
axis tight
%ylim([1.1,3.5])
%ylim([1.02,1.16])
%ylim([0.96,1.04])
set(gcf, 'Position', [0,0,1300,800]);
saveas(gcf,['norm_error_',smth,sh,'_',method],'png');
hold off;

errorbar(ps,median(abs(mD1./x(1)*100)),median(abs(mD1./x(1)*100))-quantile(abs(mD1./x(1)*100),.25),quantile(abs(mD1./x(1)*100),.75)-median(abs(mD1./x(1)*100)),'k','Linewidth', 5);
hold on;
errorbar(ps,median(abs(mD2./x(2)*100)),median(abs(mD2./x(2)*100))-quantile(abs(mD2./x(2)*100),.25),quantile(abs(mD2./x(2)*100),.75)-median(abs(mD2./x(2)*100)),'--k','Linewidth', 5);
errorbar(ps,median(abs(mD1_2./x(1)*100)),median(abs(mD1_2./x(1)*100))-quantile(abs(mD1_2./x(1)*100),.25),quantile(abs(mD1_2./x(1)*100),.75)-median(abs(mD1_2./x(1)*100)),'r','Linewidth', 3);
errorbar(ps,median(abs(mD2_2./x(2)*100)),median(abs(mD2_2./x(2)*100))-quantile(abs(mD2_2./x(2)*100),.25),quantile(abs(mD2_2./x(2)*100),.75)-median(abs(mD2_2./x(2)*100)),'--r','Linewidth', 3);
set(gcf, 'Position', [0,0,1300,800]);
%plot(ps,mD./x*100,'Linewidth', 3);
%legend('error 1','error 2','error 3')
set(gca,'FontSize',40)
xlabel('$\hat r^+$', 'Interpreter','latex');
ylabel('Error rate (%)','Interpreter','latex')
axis tight
ylim([0,25])
xlim([8,18])
saveas(gcf,['d_error_',smth,method],'png');
hold off;

errorbar(ps,median(abs(mA1)),median(abs(mA1))-quantile(abs(mA1),.25),quantile(abs(mA1),.75)-median(abs(mA1)),'b','Linewidth', 5);
hold on;
errorbar(ps,median(abs(mA2)),median(abs(mA2))-quantile(abs(mA2),.25),quantile(abs(mA2),.75)-median(abs(mA2)),'--b','Linewidth', 5);
errorbar(ps,median(abs(mA1_2)),median(abs(mA1_2))-quantile(abs(mA1_2),.25),quantile(abs(mA1_2),.75)-median(abs(mA1_2)),'r','Linewidth', 5);
errorbar(ps,median(abs(mA2_2)),median(abs(mA2_2))-quantile(abs(mA2_2),.25),quantile(abs(mA2_2),.75)-median(abs(mA2_2)),'--r','Linewidth', 5);
%legend('error 1','error 2','error 3')
set(gca,'FontSize',20)
xlabel('n');
ylabel('Angle Error')
axis tight
ylim([0,0.5])
saveas(gcf,['a_error_',smth,method],'png');
hold off;



plot(ps, sum(mrSN==2)./T*100,'g','Linewidth', 5);
hold on;
plot(ps, sum(mrSN==1)./T*100,':g','Linewidth', 5);
plot(ps,sum(mrp==2)./T*100,'k','Linewidth', 5);
plot(ps,sum(mrp==1)./T*100,':k','Linewidth', 5);
set(gca,'FontSize',20)
xlabel('n');
ylabel('%')
axis tight
%set(gcf, 'Position', [0,0,1300,800]);
saveas(gcf,['rank_',smth ,method],'png');
hold off;
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