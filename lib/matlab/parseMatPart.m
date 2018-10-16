function B=parseMatPart(A,szSub,ovSub);
%%%
%%% Works like parseMatFull, but includes extra overlap command, which
%%% contains integer values that specify the amount of shift between
%%% windows.


szSub = szSub(:)';
ovSub = ovSub(:)';

szA = ones(1,3);
ovA = ones(3,1);
tmp = size(A);
szA(1:length(tmp)) = tmp;
ndSub = length(szSub);

if length(szA)<ndSub;
  error('Dimensions of submatrices must be less than or equal to original matrix');
end
if any(szA(1:ndSub)<szSub);
  error('Size of submatrices cannot exceed size of original matrix')
end

szA(1:ndSub) = ceil((szA(1:ndSub)-szSub+1)./ovSub);
nA = prod(szA);
indx = ones(3,1);
indx(1:ndSub) = szSub;
ovA(1:ndSub) = ovSub;

B = zeros(prod(szSub),nA);
cnt=1;


for ii=1:szA(3);
  for jj=1:szA(2);
    for kk=1:szA(1);
      tmp = A((1:indx(1))+(kk-1)*ovA(1),(1:indx(2))+(jj-1)*ovA(2),...
      					(1:indx(3))+(ii-1)*ovA(3));
      B(:,cnt) = tmp(:);
      cnt=cnt+1;
    end
  end
end
