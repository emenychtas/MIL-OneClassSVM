function [Trecon,varargout] = kruskal_reconstruct(Tkruskal,varargin)
    % This function takes the output of a CPD in the form of a cell containing
    % kruskal matrices and outputs the reconstructed tensor. Also, in case the
    % original tensor T is provided this function outputs the relative error
    % of the reconstruction as well.
    %
    % Usage:
    %
    % [Trecon] = kruskal_reconstruct(Tkruskal);
    %
    % OR
    %
    % [Trecon,Trelerr] = kruskal_reconstruct(Tkruskal,T);
    
    Trecon = cpdgen(Tkruskal);
    if nargout==2 && nargin==2
        varargout{1} = frob(cpdres(varargin{1},Tkruskal))/frob(varargin{1});
    end
end