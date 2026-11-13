function out=tensorprod(a,b,axis_a,axis_b)
% Narrow Octave compatibility adapter for official HR.fft's contraction.
% It implements only (n,n,nR) times (nR,1) over axes (3,1), not the full
% MATLAB tensorprod API. The official Fourier/derivative code is unmodified.
if axis_a~=3 || axis_b~=1 || size(a,3)~=numel(b)
    error('fixture tensorprod supports only the HR.fft contraction');
end
out=reshape(reshape(a,[],size(a,3))*b(:),size(a,1),size(a,2));
end
