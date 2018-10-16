function sigOut = nonIterLsInvStft(in);

N=in.N;
K=in.K;
S=round(N*(1-in.fracOvrlp));
M=size(in.stft,2); %ceil((sigLen-N+1)/S);
J=(M-1)*S+N;

%%%%%%%%%%%%%%%%%%%%%
%%% CREATE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%%
if length(in.winInfo)==1;
  wVec = window(cell2mat(in.winInfo(1)),in.N);
elseif length(in.winInfo)==2;
  wVec = window(cell2mat(in.winInfo(1)),in.N, cell2mat(in.winInfo(2)));
elseif length(in.winInfo)==3;
  wVec = window(cell2mat(in.winInfo(1)),N, cell2mat(in.winInfo(2)), cell2mat(in.winInfo(3)));
else
  error('Window information for the Short-Time Fourier Transform improperly defined');
end

wVecSq = wVec.^2;

%%% STFT Parsing
vecC = 1:S:J;
vecC = vecC((vecC+in.N-1)<=in.origSigSize(1));

%%% Normalization Array
DlsArr = zeros([in.origSigSize(1),1]);
for jj=1:length(vecC)
    tmpArr = vecC(jj):(vecC(jj)+N-1);
    DlsArr(tmpArr) = DlsArr(tmpArr)+wVecSq;
end
DlsArrInv = 1./DlsArr;

%%% Inverse STFT
invFT = sqrt(in.N)*ifft(in.stft);
invFT = invFT(1:in.N,:,:,:).*repmat(wVec,[1,size(invFT,2),size(invFT,3),size(invFT,4)]);

%%% Recombine segments
yEst = zeros([in.origSigSize]);
for kk=1:length(vecC);
    tmpArr = vecC(kk):(vecC(kk)+N-1);
    yEst(tmpArr,:) = yEst(tmpArr,:)+squeeze(invFT(:,kk,:));
end

%%% Normalize Recombined Segments
sigOut = yEst.*repmat(DlsArrInv,[1,size(yEst,2),size(yEst,3)]);
