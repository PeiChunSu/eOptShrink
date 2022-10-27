%clear; close all;
cd('C:\Users\NTU_Math\Documents\prelim\color noise code')
x = [4,4,4,2.2,2.2,2.2, 1.5,1.5,1.5,0.4,0.4,0.4];
r = length(x);
%gamma = 1; name1 = 'Unif[1,10]';name2 = 'Mix2.2'; r_eff = 6;
%gamma = 0.5;name1 = 'Unif[1,10]';name2 = 'Mix2.2'; r_eff = 6;
%gamma = 1;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur'; r_eff = 9;
gamma = 0.5;name1 = 'Marcenko-Pastur';name2 = 'Marcenko-Pastur'; r_eff = 9;
%9;
r = length(x);
cm = 10; % student t 10 's moment
c = 2;
%kk = 4; % k for imputation
smth = '';

%name1 = 'Marcenko-Pastur';
%name1 = 'Mix2';
%name1 = 'Unif[1,10]';
%name1 = 'PaddedIdentity';
%name1 = 'Chi10';
%name1 = 'Fisher3n';
%name1 = 'Triangle';



%name2 = 'Marcenko-Pastur';
%name2 = 'Mix2';
%name2 = 'Mix2.2';
%name2 = 'Unif[1,10]';
%name2 = 'PaddedIdentity';
%name2 = 'Chi10';
%name2 = 'Fisher3n';
%name2 = 'Triangle';
method = 'imp';
sh = 'op2';

if sh == "fro"
    err_method = 'fro';
elseif sh == "op"
    err_method = 2;

end
ps = 2000;%200:200:800;
T = 1%50;

mF = [];mO = [];mN = [];
sF = [];sO = [];sN = [];
mF0 = [];mO0 = [];mN0 = [];
sF0 = [];sO0 = [];sN0 = [];
mS0 = [];mSw = [];mSi = [];
sS0 = [];sSw = [];sSi = [];
mwF = [];mwO = [];mwN = [];
swF = [];swO = [];swN = [];
mD1 = [];mA1 = [];mD2 = [];mA2 = [];
mrSN = [];mrp = [];mrold = [];mrW = [];
mDSN1 = [];mASN1 = [];mDSN2 = [];mASN2 = [];
mDW1 = [];mAW1 = [];mDW2 = [];mAW2 = [];
mD01 = [];mA01 = [];mD02 = [];mA02 = [];
er_cc= [];
for i = 1:length(ps)
    n = ps(i);
    errF = [];errO = [];errN = [];errD = [];
    errF0 = [];errO0 = [];errN0 = [];
    err_S0 = [];err_Sw = [];err_Si = [];
    err_wF = []; err_wO = []; err_wN = [];
    errA = []; err_rSN = [];err_rp = [];
    errDSN = []; errDW = []; errD0 = [];
    for t  = 1:T

        p = floor(n*gamma);
        G = randn(n,n);
        [V0, ~] =qr(G);
        V0 = V0(:,1:r);

        G = randn(p,p);
        [U0, ~] = qr(G);
        U0 = U0(:,1:r);

        X = U0 * diag(x) * V0';
        N = trnd(cm,p,n)./(sqrt(1.25)*sqrt(n));
        %student t 10/poisson mean 0/

        %[Q1,~] = qr(randn(p,p));% modified QR decomposition
        Q1 = diag(ones(1,p));%
        %[Q2,~] = qr(randn(n,n));
        Q2 = diag(ones(1,n));%
        [~,fT1] = Noise(p,p,name1,Q1);
        [~,fT2] = Noise(n,n,name2,Q2);
        A = Q1*fT1*Q1';
        B = Q2*fT2*Q2';
        sigma1 = sqrt(sum(diag(fT1).^2)/p);
        sigma2 = sqrt(sum(diag(fT2).^2)/n);
        A = A./sigma1; B = B./sigma2;
        Y  = X + A*N*B;
      
        [p,n] = size(Y);
        bulk = 1+sqrt(p/n);
        [U,s,V] = svd(Y);
        s = diag(s);
        lab = eig(Y*Y'); lab = sort(lab,'descend');
        k = floor(p^(1/2.01));
        lambda_p = lab(k+1)+ 1.1/(2^(2/3)-1) * (lab(k+1)-lab(2*k+1));
        err_lambda_p = abs(sqrt(lambda_p)- bulk)/bulk*100;
        errF = [errF; err_lambda_p];

    end
    mF = [mF, errF];sF = [sF, std(errF)];


end
errorbar(ps,median(mF),median(mF)-quantile(mF,.25),quantile(mF,.75)-median(mF),'k','Linewidth', 5);

set(gca,'FontSize',20)
xlabel('n');
ylabel('Error rate (%)')
axis tight

ylim([0,20])
%ylim([0.96,1.04])
%set(gcf, 'Position', [0,0,1300,800]);
saveas(gcf,['labp_error_',smth,sh,'_',method],'png');
hold off;

[p,n] = size(Y);
transpose = 0;
if p>n
    Y = Y';
    transpose = 1;
end
%{
[p,n] = size(Y);
[U,s,V] = svd(Y);
s = diag(s);



u = eig(Y'*Y); u = sort(u,'descend');
lab = eig(Y*Y'); lab = sort(lab,'descend');

k = floor(p^(1/2.01));
fZ = createPseudoNoise(s, k, 'i');
lambda_p = lab(k+1)+ 1.1/(2^(2/3)-1) * (lab(k+1)-lab(2*k+1));
histogram(s,100,'Normalization','Probability')
xlabel('Singular Values');
ylabel('Probability Density')
xline(sqrt(lambda_p),'LineWidth',2,'Color','r')
set(gca,'FontSize',20)
axis tight
%}