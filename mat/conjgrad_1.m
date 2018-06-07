function [xs, is] = conjgrad_1( Afunc, b, x0, maxiters, miniters, Mdiag, jacket )

if jacket == 1
    makeDouble = @(x) double(gather(x));
else
    makeDouble = @double;
end

tolerance = 5e-4;

gapratio = 0.1;
mingap = 10;

maxtestgap = max(ceil(maxiters * gapratio), mingap) + 1;

vals = zeros(maxtestgap,1);

inext = 5;
imult = 1.3;

is = [];
xs = {};

r = Afunc(x0) - b;
y = r./Mdiag;

p = -y;
x = x0;

%val is the value of the quadratic model
val = 0.5*makeDouble((-b+r)'*x);
%disp( ['iter ' num2str(0) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', val = ' num2str( val ) ]);

for i = 1:maxiters
    
    %compute the matrix-vector product.  This is where 95% of the work in
    %HF lies:
    Ap = Afunc(p);
    
    pAp = p'*Ap;

    %the Gauss-Newton matrix should never have negative curvature.  The
    %Hessian easily could unless your objective is convex
    if pAp <= 0
        disp('Negative Curvature!');
        
        disp('Bailing...');
        break;
    end
    
    alpha = (r'*y)/pAp;
    
    x = x + alpha*p;
    r_new = r + alpha*Ap;
    
    y_new = r_new./Mdiag;
    
    beta = (r_new'*y_new)/(r'*y);
    
    p = -y_new + beta*p;

    r = r_new;
    y = y_new;

    
    val = 0.5*makeDouble((-b+r)'*x);
    vals( mod(i-1, maxtestgap)+1 ) = val;
    
    %disp( ['iter ' num2str(i) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', val = ' num2str( val ) ]);
    
    testgap = max(ceil( i * gapratio ), mingap);
    prevval = vals( mod(i-testgap-1, maxtestgap)+1 ); %testgap steps ago

    if i == ceil(inext)
        is(end+1) = i;
        xs{end+1} = x;
        inext = inext*imult;
    end    
    
    %the stopping criterion here becomes largely unimportant once you
    %optimize your function past a certain point, as it will almost never
    %kick in before you reach i = maxiters.  And if the value of maxiters
    %is set so high that this never occurs, you probably have set it too
    %high
    if i > testgap && prevval < 0 && (val - prevval)/val < tolerance*testgap && i >= miniters
        break;
    end    
    
    
end

if i ~= ceil(inext)
    is(end+1) = i;
    xs{end+1} = x;
end
