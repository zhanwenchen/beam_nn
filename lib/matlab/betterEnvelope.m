function envFilt = betterEnvelope(rfIn)

%  cutoff == 0.2 and 0.8 from  fs = 20.832 MHz
%  f_cutoff = 0.2*(fs/2) = 2.0832MHz, 0.8*(fs/2) = 8.3328,
%  f0 = fs/4 = 5.208MHz fixed in Verasonics US systems

B=firpm(10,[.2 .8],[1 1],'Hilbert');

Q = filtfilt(B,1,rfIn);

env = sqrt(rfIn(:).^2 + Q(:).^2);

%[B,A]=butter(5,.4,'low'); % original filter params

% tune run_2
[B,A]=butter(5,.25,'low');


envFilt = filtfilt(B,A,env);

envFilt(envFilt<0) = 0;
