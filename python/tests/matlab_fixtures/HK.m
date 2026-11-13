classdef HK
% Minimal degree-one coefficient container for executing actual applyRU.m.
% It supplies storage/page algebra, not an alternate symmetry algorithm.
properties
    HnumL
    HcoeL
    Degree = 1
end
methods
    function obj=HK(h)
        obj.HnumL=h;obj.HcoeL=zeros(size(h));
    end
    function mats=matgen(obj,r)
        mats={1,r};
    end
end
methods(Static)
    function indices=orderlist(degree)
        if degree==0
            indices=1;
        elseif degree==1
            indices=2:4;
        else
            error('fixture supports degrees zero and one only');
        end
    end
    function values=factorlist_parity(degree)
        if degree~=1;error('fixture supports degree one');end;
        values=[1,-1,-1,-1];
    end
    function out=matrixtimespage(mat,page)
        out=zeros(size(page));
        if isvector(mat)
            for i=1:numel(mat);out(:,:,i)=mat(i)*page(:,:,i);end;
            return;
        end
        for i=1:size(mat,1)
            for j=1:size(mat,2)
                out(:,:,i)=out(:,:,i)+mat(i,j)*page(:,:,j);
            end
        end
    end
end
end
