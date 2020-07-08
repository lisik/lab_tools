function b = banded_ridge(y,X1,X2,k1,k2,flag)
%Modified Matlab RIDGE Ridge regression function to do banded ridge
%regression
%Leyla Isik 7/8/20
%   B1 = BANDED_RIDGE(Y,X1,X2,K1,K2) returns the vector B1 of regression coefficients
%   obtained by performing ridge regression of the response vector Y
%   on the predictors X using ridge parameter K.  The matrix X should
%   not contain a column of ones.  The results are computed after
%   centering and scaling the X columns so they have mean 0 and
%   standard deviation 1.  If Y has n observations, X is an n-by-p
%   matrix, and K1 is a scalar, the result B1 is a column vector with p
%   elements.  If K1 has m elements, B1 is p-by-m.
%   Currently only supports one-dimensional K2.
%
%   B0 = RIDGE(Y,X1,X2,K1,K2,0) performs the regression without centering and
%   scaling.  The result B0 has p+1 coefficients, with the first being
%   the constant term.   RIDGE(Y,X1,X2,K1,K2,1) is the same as RIDGE(Y,X1,X2,K1,K2).
%


%   Copyright 1993-2008 The MathWorks, Inc.


if nargin < 5,              
    error(message('Requires at least five input arguments'));      
end 

if nargin<6 || isempty(flag) || isequal(flag,1)
   unscale = false;
elseif isequal(flag,0)
   unscale = true;
else
   error(message('stats:ridge:BadScalingFlag'));
end

% Check that matrix (X) and left hand side (y) have compatible dimensions
[n1,p1] = size(X1);
[n2,p2] = size(X2);

if p1~=p2
    error(message('X1 and X2 must have the same number of rows'))
end

[n,collhs] = size(y);
if n1~=n 
    error(message('stats:ridge:InputSizeMismatch')); 
end 
if n2~=n 
    error(message('stats:ridge:InputSizeMismatch')); 
end 

if collhs ~= 1, 
    error(message('stats:ridge:InvalidData')); 
end

% Remove any missing values
wasnan = (isnan(y) | any(isnan([X1 X2]),2));
if (any(wasnan))
   y(wasnan) = [];
   X1(wasnan,:) = [];
   X2(wasnan,:) = [];
   n = length(y);
end

% Normalize the columns of X1 to mean zero, and standard deviation one.
mx1 = mean(X1);
stdx1 = std(X1,0,1);
idx1 = find(abs(stdx1) < sqrt(eps(class(stdx1)))); 
if any(idx1)
  stdx(idx1) = 1;
end

MX1 = mx1(ones(n1,1),:);
STDX1 = stdx1(ones(n1,1),:);
Z1 = (X1 - MX1) ./ STDX1;
if any(idx1)
  Z1(:,idx1) = 1;
end

% Normalize the columns of X2 to mean zero, and standard deviation one.
mx2 = mean(X2);
stdx2 = std(X2,0,1);
idx2 = find(abs(stdx2) < sqrt(eps(class(stdx2)))); 
if any(idx2)
  stdx(idx2) = 1;
end

MX2 = mx2(ones(n2,1),:);
STDX2 = stdx2(ones(n2,1),:);
Z2 = (X2 - MX2) ./ STDX2;
if any(idx2)
  Z2(:,idx2) = 1;
end

% Compute the ridge coefficient estimates using the technique of
% adding pseudo observations having y=0 and X'X = k*I.
pseudo = [sqrt(k1(1)) * eye(p1) zeros(p1); zeros(p2) sqrt(k2(1)) * eye(p2)];
%pseudo2 = sqrt(k2(1)) * eye(p2);
Zplus  = [Z1 Z2;pseudo];
yplus  = [y;zeros(p1,1); zeros(p2,1)];

% Compute the coefficient estimates
b = Zplus\yplus;

% LI Note: Currently only works for multiple k1 and single k2
% Set up an array to hold the results
nk1 = numel(k1);
nk2 = numel(k2);

if nk1>1
   % Fill in more entries after first expanding b.  We did not pre-
   % allocate b because we want the backslash above to determine its class.
   b(end,nk1) = 0;
   for j=2:nk1
      Zplus(end-(p1+p2)+1:end,:) = [sqrt(k1(j)) * eye(p1) zeros(p1); zeros(p2) sqrt(k2(1)) * eye(p2)];
      b(:,j) = Zplus\yplus;
   end
end

% Put on original scale if requested
if unscale
   b = b ./ repmat([stdx1 stdx2]',1,nk1);
   b = [mean(y(1:length(y)/2))-[mx1 mx2]*b; b];
end
