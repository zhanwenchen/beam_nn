%function stft;
%winInfo = {@tukeywin};
function out = stft(signal, N, fracOvrlp, padding, winInfo);


if 0
%%% Test signal (no noise)
fs = 40e6;
f0 = 1.25e6;
bw = .25;
tc = gauspuls('cutoff',f0,bw,[],-40);
tGaus = -tc:(1/fs):tc;
puls = gauspuls(tGaus,f0,bw);
scats = randn(6000,64);
testSig = conv2(scats, puls(:),'same');
signal = testSig;
disp('Done Simulating signal')
end


%%%%
% M = number of sections
% N = length of each section
% S = shift length (samples)
% J = (M-1)S + N... which is what the total new elements?!?!

%Ovec is a matrix that creates a decimated version of parseMatFull

%w(n) is the window function for time to frequency
%can be zero padded

% K = length of fourier transformed signal
%N = 256;
S = round(N*(1-fracOvrlp));
if length(padding) ==0;
padding = N;
end
if padding<N;
  padding = N;
end
K=padding;


%%% Minus the spatial information the stft can be done as...
zeroExist =1;
zeroCrct = 0;

while zeroExist==1;
if length(winInfo) == 0;
  wVec = tukeywin(N+zeroCrct);
  winInfo = {@tukeywin};
elseif length(winInfo)==1;
  wVec = window(cell2mat(winInfo(1)),N+zeroCrct);
elseif length(winInfo)==2;
  wVec = window(cell2mat(winInfo(1)),N+zeroCrct, cell2mat(winInfo(2)));
elseif length(winInfo)==3;
  wVec = window(cell2mat(winInfo(1)),N+zeroCrct, cell2mat(winInfo(2)), cell2mat(winInfo(3)));
else
  error('Window information for the Short-Time Fourier Transform improperly defined');
end


wVec = wVec((1+zeroCrct/2):(end-zeroCrct/2));

if any(wVec==0);
  zeroCrct = sum(wVec==0);
else
  zeroExist = 0;
end

end

szSignal = size(signal);
%allOvrlp = parseMatFull(signal, [N,1]);
%newShape = [N,size(allOvrlp,2)/prod(szSignal(2:end)),szSignal(2:end)];
%allOvrlp = reshape(allOvrlp,newShape); %N,[],szSignal(2:end)); %size(signal,2));
%subOvrlp = allOvrlp(:,1:S:end,:,:);
allOvrlp = parseMatPart(signal, [N,1,1],[S,1,1]);
newShape = [N,size(allOvrlp,2)/prod(szSignal(2:end)),szSignal(2:end)];
subOvrlp = reshape(allOvrlp,newShape); %N,[],szSignal(2:end)); %size(signal,2));
%subOvrlp = allOvrlp(:,1:S:end,:,:);
%startLocs = 1:S:size(allOvrlp,2);
startLocs = 1:S:size(signal,1);
startLocs = startLocs(1:size(subOvrlp,2));


szSubOvrlp = size(subOvrlp);

winOvrlp = subOvrlp.*repmat(wVec,[1,szSubOvrlp(2:end)] ); %size(subOvrlp,2), size(subOvrlp,3)]);

%winPad = [winOvrlp]; % zeros([K-N, size(winOvrlp,2)])];
stft = fft(winOvrlp,K);
freq = ((0:(K-1))/K);

%keyboard;

out.stft = stft;
out.freqs = freq;
out.startOffsets = startLocs;
out.N = N;
out.K = K;
out.winInfo = winInfo;
out.fracOvrlp = fracOvrlp;
