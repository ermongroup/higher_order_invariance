function paramsp = nnet_train_geo_faster( thresh, runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket)
% This is the real faster one! need to think more about back-tracking and
% rate choosing.
%
% Demo code for the paper "Deep Learning via Hessian-free Optimization" by James Martens.
%
% paramsp = nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket)
%
% IMPORTANT NOTES:  The most important variables to tweak are `initlambda' (easy) and
% `maxiters' (harder).  Also, if your particular application is still not working the next 
% most likely way to fix it is tweaking the variable `initcoeff' which controls
% overall magnitude of the initial random weights.  Please don't treat this code like a black-box,
% get a negative result, and then publish a paper about how the approach doesn't work :)  And if
% you are running into difficulties feel free to e-mail me at james.martens@gmail.com
%
% runName - name you give to the current run.  This is used for the
% log-file and the files which contain the current parameters that get
% saved every 10 epochs
%
% runDesc - notes to yourself about the current run (can be the empty string)
%
% paramsp - initial parameters in the form of a vector (can be []).  If
% this, or the arguments Win,bin are empty, the 'sparse initialization'
% technique is used
%
% Win, bin - initial parameters in their matrix forms (can be [])
%
% resumeFile - file used to resume a previous run from a file
%
% maxepoch - maximum number of 'epochs' (outer iteration of HF).  There is no termination condition
% for the optimizer and usually I just stop it by issuing a break command
%
% indata/outdata - input/output training data for the net (each case is a
% column).  Make sure that the training cases are randomly permuted when you invoke
% this function as it won't do it for you.
%
% numchunks - number of mini-batches used to partition the training set.
% During each epoch, a single mini-batch is used to compute the
% matrix-vector products, after which it gets cycled to the back of the
% last and is used again numchunk epochs later. Note that the gradient and 
% line-search are still computed using the full training set.  This of
% course is not practical for very large datasets, but in general you can
% afford to use a lot more data to compute the gradient than the
% matrix-vector products, since you only have to do the former once per iteration
% of the outer loop.
%
% intest/outtest -  test data
%
% numchunks_test - while the test set isn't used for matrix-vector
% products, you still may want to partition it so that it can be processed
% in pieces on the GPU instead of all at once.
%
% layersizes - the size of each hidden layer (input and output sizes aren't
% specified in this vector since they are determined by the dimension of
% the data arguements) 
%
% layertypes - a cell-array of strings that indicate the unit type in each
% layer.  can be 'logistic', 'tanh', 'linear' or 'softmax'.  I haven't
% thoroughly tested the softmax units.  Also, the output units can't be
% tanh because I haven't implemented that (even though it's easy).
% Consider that an exercise :)
%
% mattype - the type of curvature matrix to use.  can be 'gn' for
% Gauss-Newton, 'hess' for Hessian and 'empfish' for empirical Fisher.  You
% should probably only ever use 'gn' if you actually want the training to
% go well
%
% rms - by default we use the canonical error function for
% each output unit type.  e.g. square error for linear units and
% cross-entropy error for logistics.  Setting this to 1 (instead of 0) overrides 
% the default and forces it to use squared-error.  Note that even if what you
% care about is minimizing squared error it's sometimes still better
% to run on the optimizer with the canonical error
%
% errtype - in addition to displaying the objective function (log-likelihood) you may also
% want to keep track of another metric like squared error when you train
% deep auto-encoders.  This can be 'L2' for squared error, 'class' for
% classification error, or 'none' for nothing.  It should be easy enough to
% add your own type of error should you need one
%
% hybridmode - set this 1 unless you want compute the matrix-vector
% products using the whole training dataset instead of the mini-batches.
% Note that in this case they still serve a purpose since the mini-batches
% are only loaded onto the gpu 1 at a time.
%
% weightcost - the strength of the l_2 prior on the weights
%
% decay - the amount to decay the previous search direction for the
% purposes of initializing the next run of CG.  Should be 0.95
%
% jacket - set to 1 in order to use the Jacket computing library.  Will run
% on the CPU otherwise and hence be really slow.  You can easily port this code
% over to free and possibly slower GPU packages like GPUmat (in fact, I have some
% commented code which does just that (do a text search for "GPUmat version")


disp( ['Starting run named: ' runName ]);

rec_constants = {'layersizes', 'rms', 'weightcost', 'hybridmode', 'autodamp', 'initlambda', 'drop', 'boost', 'numchunks', 'mattype', 'errtype', 'decay'};


autodamp = 1;

drop = 2/3;

boost = 1/drop;


%In addition to maxiters the variable below is something you should manually
%adjust.  It is quite problem specific.  Fortunately after only 1 'epoch'
%you can often tell if you've made a bad choice.  The value of rho should lie
%somewhere between 0.75 and 0.95.  I could automate this part but I'm lazy
%and my code isn't designed to make such automation a natural thing to add.  Also
%note that 'lambda' is being added to the normalized curvature matrix (i.e.
%divided by the number of cases) while in the ICML paper I was adding it to
%the unnormalized curvature matrix.  This doesn't make any real
%difference to the optimization, but does make it somewhat easier to guage
%lambda and set its initial value since it will be 'independent' of the
%number of training cases in each mini-batch
initlambda = 45.0;

if rms ~= 0
    error('rms must be 0!')
end

if strcmp(mattype, 'hess')
    storeD = 1;
    computeBV = @computeHV;
elseif strcmp(mattype, 'gn')
    storeD = 0;
    computeBV = @computeGV;
elseif strcmp(mattype, 'empfish')
    storeD = 0;
    computeBV = @computeFV;
end



% Hack here
% Use matlab gpu support instead of jacket
% since jacket is never available any more
if jacket
    %{
    mones = @gones;
    mzeros = @gzeros;
    conv = @gsingle;
    
    %GPUmat version:
    %mones = @(varargin) ones(varargin{:}, GPUsingle);
    %mzeros = @(varargin) zeros(varargin{:}, GPUsingle);
    %conv = @GPUsingle;
    
    %norm = @(x) sqrt(sum(x.*x));
    
    mrandn = @grandn;
    %}
    mones = @(varargin) ones(varargin{:}, 'single', 'gpuArray');
    mzeros = @(varargin) zeros(varargin{:}, 'single', 'gpuArray');
    conv = @gpuArray;
    
    %GPUmat version:
    %mones = @(varargin) ones(varargin{:}, GPUsingle);
    %mzeros = @(varargin) zeros(varargin{:}, GPUsingle);
    %conv = @GPUsingle;
    
    %norm = @(x) sqrt(sum(x.*x));
    
    mrandn = @(varargin) randn(varargin{:}, 'single', 'gpuArray');
    makeDouble = @(x) double(gather(x));
else
    %use singles (this can make cpu code go faster):
    
    mones = @(varargin)ones(varargin{:}, 'single');
    mzeros = @(varargin)zeros(varargin{:}, 'single');
    %conv = @(x)x;
    conv = @single;
    
    
    %use doubles:
    %{
    mones = @ones;
    mzeros = @zeros;
    %conv = @(x)x;
    conv = @double;
    %}
    
    mrandn = @randn;
    makeDouble = @double;
end

if hybridmode
    store = conv; %cache activities on the gpu

    %store = @single; %cache activities on the cpu
else
    store = @single;
end


layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims numcases] = size(indata);
[tmp numtest] = size(intest);

if mod( numcases, numchunks ) ~= 0
    error( 'Number of chunks doesn''t divide number of training cases!' );
end

sizechunk = numcases/numchunks;
sizechunk_test = numtest/numchunks_test;


if numcases >= 512*64
    disp( 'jacket issues possible!' );
end


y = cell(numchunks, numlayers+1);
if storeD
    dEdy = cell(numchunks, numlayers+1);
    dEdx = cell(numchunks, numlayers);
end



function v = vec(A)
    v = A(:);
end


psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

%pack all the parameters into a single vector for easy manipulation
function M = pack(W,b)
    
    M = mzeros( psize, 1 );
    
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
    end
    
end

%unpack parameters from a vector so they can be used in various neural-net
%computations
function [W,b] = unpack(M)

    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
    
    cur = 0;
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );

        cur = cur + layersizes(i)*layersizes(i+1);
        
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );

        cur = cur + layersizes(i+1);
    end
    
end


%compute the vector-product with the Hessian matrix
function HV = computeHV(V)
    
    if ~storeD
        error('need to store D');
    end

    [VWu, Vbu] = unpack(V);
    
    HV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    
    for chunk = chunkrange

        %application of R operator
        Ry = cell(numlayers+1,1);
        RdEdy = cell(numlayers+1,1);
        RdEdx = cell(numlayers, 1);
        HVW = cell(numlayers,1);
        HVb = cell(numlayers,1);

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
        yip1 = conv(y{chunk, 1});
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);

            Ry{i} = store(Ryi);
            Ryi = [];

            yip1 = conv(y{chunk, i+1});
            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
            else
                error( 'Unknown/unsupported layer type' );
            end

            Rxi = [];

        end
   
        %backward prop:
        %cross-entropy for logistics:
        %RdEdy{numlayers+1} = (-(outdata./(y{numlayers+1}.^2) + (1-outdata)./(1-y{numlayers+1}).^2)).*Ry{numlayers+1};
        %cross-entropy for softmax:
        %RdEdy{numlayers+1} = -outdata./(y{numlayers+1}.^2).*Ry{numlayers+1};
        for i = numlayers:-1:1

            if i < numlayers
                if strcmp(layertypes{i}, 'logistic')
                    %logistics:

                    dEdyip1 = conv( dEdy{chunk, i+1} );
                    RdEdx{i} = RdEdy{i+1}.*yip1.*(1-yip1) + dEdyip1.*Ryip1.*(1-2*yip1);
                    dEdyip1 = [];

                elseif strcmp(layertypes{i}, 'linear')
                    RdEdx{i} = RdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end

            else
                if ~rms
                    %assume canonical link functions:
                    RdEdx{i} = -Ryip1;
                    
                    %if strcmp(layertypes{i}, 'linear')
                    %    RdEdx{i} = 2*RdEdx{i};
                    %end
                else
                    dEdyip1 = 2*(conv(outdata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) )) - yip1); %mult by 2 because we dont include the 1/2 before
                    RdEdyip1 = -2*Ryip1;
                    
                    if strcmp( layertypes{i}, 'softmax' )
                        %softmax:
                        RdEdx{i} = RdEdyip1.*yip1 - yip1.*repmat( sum( RdEdyip1.*yip1, 1), [layersizes(i+1) 1] ) ...
                                + dEdyip1.*Ryip1 - yip1.*repmat( sum( dEdyip1.*Ryip1, 1), [layersizes(i+1) 1] ) - Ryip1.*repmat( sum( dEdyip1.*yip1, 1), [layersizes(i+1) 1] );
                            
                        %error( 'RMS error not supported with softmax output' );
                        
                    elseif strcmp( layertypes{i}, 'logistic' )
                        RdEdx{i} = RdEdyip1.*yip1.*(1-yip1) + dEdyip1.*Ryip1.*(1-2*yip1);
                        
                    elseif strcmp(layertypes{i}, 'linear')
                        RdEdx{i} = RdEdyip1;

                    else
                        error( 'Unknown/unsupported layer type' );
                    end
                    
                    dEdyip1 = [];
                    RdEdyip1 = [];
                    
                end
            end
            RdEdy{i+1} = [];

            yip1 = []; Ryip1 = [];

            yi = conv( y{chunk, i} );
            Ryi = conv( Ry{i} );        

            dEdxi = conv( dEdx{chunk, i} );

            RdEdy{i} = VWu{i}'*dEdxi + Wu{i}'*RdEdx{i};

            %(HV = RdEdW)
            HVW{i} = RdEdx{i}*yi' + dEdxi*Ryi';
            HVb{i} = sum(RdEdx{i},2);

            RdEdx{i} = []; dEdxi = [];

            yip1 = yi; yi = [];
            Ryip1 = Ryi; Ryi = [];

        end
        yip1 = []; Ryip1 = []; RdEdy{1} = [];


        HV = HV + pack(HVW, HVb);
    end

    HV = HV / conv(numcases);
    
    if hybridmode
        HV = HV * conv(numchunks);
    end
    
    HV = HV - conv(weightcost)*(maskp.*V);
    
    if autodamp
        HV = HV - conv(lambda)*V;
    end
    
end


%compute the vector-product with the Gauss-Newton matrix
function GV = computeGV(V)

    [VWu, Vbu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);
        Ry = cell(numlayers,1);

        yip1 = conv(y{chunk, 1});

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{chunk, i+1});

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'tanh')
                Ryip1 = Rxi.*(1+yip1).*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            Rxi = [];

        end
        
        %Backwards pass.  This is where things start to differ from computeHV  Please note that the lower-case r 
        %notation doesn't really make sense so don't bother trying to decode it.  Instead there is a much better
        %way of thinkin about the GV computation, with its own notation, which I talk about in my more recent paper: 
        %"Learning Recurrent Neural Networks with Hessian-Free Optimization"
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    %assume canonical link functions:
                    rdEdx{i} = -Ryip1;
                    
                    %if strcmp(layertypes{i}, 'linear')
                    %    rdEdx{i} = 2*rdEdx{i};
                    %end
                else
                    RdEdyip1 = -2*Ryip1;
                    
                    if strcmp(layertypes{i}, 'softmax')
                        error( 'RMS error not supported with softmax output' );
                    elseif strcmp(layertypes{i}, 'logistic')
                        rdEdx{i} = RdEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        rdEdx{i} = RdEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        rdEdx{i} = RdEdyip1;
                    else
                        error( 'Unknown/unsupported layer type' );
                    end
                    
                    RdEdyip1 = [];
                    
                end
                
                Ryip1 = [];

            end
            rdEdy{i+1} = [];
            
            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb);
        
    end
    
    GV = GV / conv(numcases);
    
    if hybridmode
        GV = GV * conv(numchunks);
    end
    
    %GV = GV - conv(weightcost)*(maskp.*V);

    if autodamp
        GV = GV - conv(lambda)*V;
    end
    
end

function GV = term1(V)

    [VWu, Vbu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);
        Ry = cell(numlayers,1);

        yip1 = conv(y{chunk, 1});

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
        Syip1 = mzeros(layersizes(1), sizechunk);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];
            Syi = Syip1;
            Syip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);          
            %Rx{i} = store(Rxi);
            Sxi = 2 * VWu{i} * Ryi + Wu{i} * Syi;

            yip1 = conv(y{chunk, i+1});

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
                Syip1 = (2*yip1-1).*yip1.*(yip1-1).*Rxi.*Rxi + yip1.*(1-yip1).*Sxi; 
            elseif strcmp(layertypes{i}, 'tanh')
                Ryip1 = Rxi.*(1+yip1).*(1-yip1);
                Syip1 = (yip1-1).*(yip1+1).*(2*yip1).*Rxi.*Rxi + (1+yip1).*(1-yip1).*Sxi; 
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
                Syip1 = Sxi;            
            elseif strcmp( layertypes{i}, 'softmax' )
                n = layersizes(i+1);
                yip1Rxi = sum(yip1 .* Rxi, 1);
                Ryip1 = Rxi.*yip1 - yip1.* repmat( yip1Rxi, [n 1] );                
                Syip1 = yip1 .* repmat(2 * yip1Rxi.^2, [n 1]) +...
                    yip1 .* (Rxi.^2) - 2 * yip1 .* Rxi .* repmat(yip1Rxi, [n 1])...
                    - yip1 .* repmat(sum(yip1 .* Rxi.^2, 1), [n 1]) + yip1 .* Sxi...
                    - yip1 .* repmat(sum(yip1 .* Sxi, 1), [n 1]);
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            Rxi = [];
            Sxi = [];
        end
        
        %Backwards pass.  This is where things start to differ from computeHV  Please note that the lower-case r 
        %notation doesn't really make sense so don't bother trying to decode it.  Instead there is a much better
        %way of thinkin about the GV computation, with its own notation, which I talk about in my more recent paper: 
        %"Learning Recurrent Neural Networks with Hessian-Free Optimization"
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    %assume canonical link functions:
                    rdEdx{i} = Syip1; % Correct the sign
                    
     %               if strcmp(layertypes{i}, 'linear')
      %                  rdEdx{i} = 2*rdEdx{i};
       %             end
                else
                    error( 'Not supported in term1' );                                        
                end
                
                Ryip1 = [];
                Syip1 = [];
            end
            rdEdy{i+1} = [];
            
            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb);
        
    end
    
    GV = GV / conv(numcases);
    
    if hybridmode
        GV = GV * conv(numchunks);
    end
    %???? I am not sure about this
    %GV = GV - conv(weightcost)*(maskp.*V);    
end

function GV = term2(V)

    [VWu, Vbu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);
        Ry = cell(numlayers,1);

        yip1 = conv(y{chunk, 1});

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{chunk, i+1});

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'tanh')
                Ryip1 = Rxi.*(1+yip1).*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );                
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            Rxi = [];

        end
        
        %Backwards pass.  This is where things start to differ from computeHV  Please note that the lower-case r 
        %notation doesn't really make sense so don't bother trying to decode it.  Instead there is a much better
        %way of thinkin about the GV computation, with its own notation, which I talk about in my more recent paper: 
        %"Learning Recurrent Neural Networks with Hessian-Free Optimization"
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    if strcmp(layertypes{i}, 'logistic')
                        yip1 = conv(y{chunk, numlayers + 1});
                        coeff = ((2*yip1-1)./2./(yip1.*(1-yip1)));
                        coeff(Ryip1 == 0) = 0;
                        rdEdx{i} = coeff.*Ryip1.^2; % correct the sign
                    elseif strcmp(layertypes{i}, 'softmax')
                        yip1 = conv(y{chunk, numlayers + 1});
                        n = layersizes(i+1);
                        Ryip12overyip1 = Ryip1.^2./yip1;
                        rdEdx{i} = -1/2*(Ryip12overyip1 - yip1 .* repmat(sum(Ryip12overyip1, 1), [n 1]));
                    end
                else                    
                    error(' Not supported in term2!')                    
                end
                
                Ryip1 = [];

            end
            rdEdy{i+1} = [];
            
            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb);
        
    end
    
    GV = GV / conv(numcases);
    
    if hybridmode
        GV = GV * conv(numchunks);
    end
    %???? I am not sure about this!
    %GV = GV - conv(weightcost)*(maskp.*V);        
end

%compute the vector-product with the emperical Fisher matrix
function FV = computeFV(V)

    [VWu, Vbu] = unpack(V);
    
    FV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        FVW = cell(numlayers,1);
        FVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);


        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
        yip1 = conv(y{chunk, 1});
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{chunk, i+1});

            if i < numlayers
                if strcmp(layertypes{i}, 'logistic')
                    Ryip1 = Rxi.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    Ryip1 = Rxi.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    Ryip1 = Rxi;
                elseif strcmp( layertypes{i}, 'softmax' )
                    Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                dEdxi = conv(outdata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) )) - yip1;
                Ryip1 = repmat(sum(Rxi.*dEdxi, 1), [layersizes(i+1) 1]).*dEdxi;
                %Ryip1 = Rxi.*(dEdxi.^2);
                dEdxi = [];
                
                if rms
                    error('not sure if this works');
                end
            end

            Rxi = [];

        end

        %back prop:
        %cross-entropy for logistics:
        %dEdy{numlayers+1} = outdata./y{numlayers+1} - (1-outdata)./(1-y{numlayers+1});
        %cross-entropy for softmax:
        %dEdy{numlayers+1} = outdata./y{numlayers+1};
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    %assume canonical link functions:
                    rdEdx{i} = -Ryip1;
                    
                    %if strcmp(layertypes{i}, 'linear')
                    %    rdEdx{i} = 2*rdEdx{i};
                    %end
                else

                    RdEdyip1 = -2*Ryip1;
                    
                    if strcmp(layertypes{i}, 'softmax')
                        error( 'RMS error not supported with softmax output' );
                    elseif strcmp(layertypes{i}, 'logistic')
                        rdEdx{i} = RdEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        rdEdx{i} = RdEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        rdEdx{i} = RdEdyip1;
                    else
                        error( 'Unknown/unsupported layer type' );
                    end
                    
                    RdEdyip1 = [];
                    
                end
                
                Ryip1 = [];

            end
            rdEdy{i+1} = [];

            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            %standard gradient comp:
            FVW{i} = rdEdx{i}*yi';
            FVb{i} = sum(rdEdx{i},2);
            %FVb{i} = rdEdx{i}*mones(sizechunk,1);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        FV = FV + pack(FVW, FVb);
    end
    
    FV = FV / conv(numcases);
    if hybridmode
        FV = FV * conv(numchunks);
    end
    
    FV = FV + gradchunk*(gradchunk'*V);

    FV = FV - conv(weightcost)*(maskp.*V);

    if autodamp
        FV = FV - conv(lambda)*V;
    end
    
end


    
function [ll, err] = computeLL(params, in, out, nchunks, tchunk)

    ll = 0;
    
    err = 0;
    
    [W,b] = unpack(params);
    
    if mod( size(in,2), nchunks ) ~= 0
        error( 'Number of chunks doesn''t divide number of cases!' );
    end    
    
    schunk = size(in,2)/nchunks;
    
    if nargin > 4
        chunkrange = tchunk;
    else
        chunkrange = 1:nchunks;
    end
    
    for chunk = chunkrange
    
        yi = conv(in(:, ((chunk-1)*schunk+1):(chunk*schunk) ));
        outc = conv(out(:, ((chunk-1)*schunk+1):(chunk*schunk) ));

        for i = 1:numlayers
            xi = W{i}*yi + repmat(b{i}, [1 schunk]);

            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yi = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            elseif strcmp(layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
                tmp = [];
            end

        end

        if rms || strcmp( layertypes{numlayers}, 'linear' )
            
            ll = ll + makeDouble( -sum(sum((outc - yi).^2)));
            
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                
                %outc==0 and outc==1 are included in this formula to avoid
                %the annoying case where you have 0*log(0) = 0*-Inf = NaN
                %ll = ll + makeDouble( sum(sum(outc.*log(yi + (outc==0)) + (1-outc).*log(1-yi + (outc==1)))) );
                
                %this version is more stable:
                ll = ll + makeDouble(sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0))))));
                
                
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                
                ll = ll + makeDouble(sum(sum(outc.*log(yi))));
                
            end
        end
        xi = [];

        if strcmp( errtype, 'class' )
            %err = 1 - makeDouble(sum( sum(outc.*yi,1) == max(yi,[],1) ) )/size(in,2);
            err = err + makeDouble(sum( sum(outc.*yi,1) ~= max(yi,[],1) ) ) / size(in,2);
        elseif strcmp( errtype, 'L2' )
            err = err + makeDouble(sum(sum((yi - outc).^2, 1))) / size(in,2);
        elseif strcmp( errtype, 'none')
            %do nothing
        else
            error( 'Unrecognized error type' );
        end
        %err = makeDouble(   (mones(1,size(in,1))*((yi - out).^2))*mones(size(in,2),1)/conv(size(in,2))  );
        
        outc = [];
        yi = [];
    end

    ll = ll / size(in,2);
    
    if nargin > 4
        ll = ll*nchunks;
        err = err*nchunks;
    end
    
    ll = ll - 0.5*weightcost*makeDouble(params'*(maskp.*params));

end


function yi = computePred(params, in) %for checking G computation using finite differences
    
    [W, b] = unpack(params);
    
    yi = in;
        
    for i = 1:numlayers
        xi = W{i}*yi + repmat(b{i}, [1 size(in,2)]);
        
        if i < numlayers
            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yi = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            elseif strcmp(layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
                tmp = [];
            end
        else
            yi = xi;
        end
        
    end
end



maskp = mones(psize,1);
[maskW, maskb] = unpack(maskp);
disp('not masking out the weight-decay for biases');
for i = 1:length(maskb)
    %maskb{i}(:) = 0; %uncomment this line apply the l_2 only to the connection weights and not the biases
end
maskp = pack(maskW,maskb);


indata = single(indata);
outdata = single(outdata);
intest = single(intest);
outtest = single(outtest);


function outputString( s )
    fprintf( 1, '%s\n', s );
    fprintf( fid, '%s\r\n', s );
end



fid = fopen( [runName '.txt'], 'a' );

outputString( '' );
outputString( '' );
outputString( '==================== New Run ====================' );
outputString( '' );
outputString( ['Start time: ' datestr(now)] );
outputString( '' );
outputString( ['Description: ' runDesc] );
outputString( '' );


ch = mzeros(psize, 1);
delta = mzeros(psize, 1);

if ~isempty( resumeFile )
    outputString( ['Resuming from file: ' resumeFile] );
    outputString( '' );
    
    load( resumeFile );
    
    ch = conv(ch);
    delta = conv(delta);
    epoch = epoch + 1;
else
    
    lambda = initlambda;
    
    llrecord = zeros(maxepoch,2);
    errrecord = zeros(maxepoch,2);
    lambdarecord = zeros(maxepoch,1);
    times = zeros(maxepoch,1);
    
    totalpasses = 0;
    epoch = 1;
    
end

if isempty(paramsp)
    if ~isempty(Win)
        paramsp = pack(Win,bin);
        clear Win bin
    else
        
        %SPARSE INIT:
        paramsp = zeros(psize,1); %not mzeros
        
        [Wtmp,btmp] = unpack(paramsp);
        
        numconn = 15;
        
        for i = 1:numlayers
 
            initcoeff = 1;

            if i > 1 && strcmp( layertypes{i-1}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            if strcmp( layertypes{i}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            
            if strcmp( layertypes{i}, 'tanh' )
                btmp{i}(:) = 0.5;
            end
            
            %outgoing
            %{
            for j = 1:layersizes(i)
                idx = ceil(layersizes(i+1)*rand(1,numconn));
                Wtmp{i}(idx,j) = randn(numconn,1)*coeff;
            end
            %}
            
            %incoming
            for j = 1:layersizes(i+1)
                idx = ceil(layersizes(i)*rand(1,numconn));
                Wtmp{i}(j,idx) = randn(numconn,1)*initcoeff;
            end
            
        end
        
        
        
        paramsp = pack(Wtmp, btmp);
        
        clear Wtmp btmp
    end
    
elseif size(paramsp,1) ~= psize || size(paramsp,2) ~= 1
    error( 'Badly sized initial parameter vector.' );
else
    paramsp = conv(paramsp);
end

outputString( 'Initial constant values:' );
outputString( '------------------------' );
outputString( '' );
for i = 1:length(rec_constants)
    outputString( [rec_constants{i} ': ' num2str(eval( rec_constants{i} )) ] );
end

outputString( '' );
outputString( '=================================================' );
outputString( '' );


for epoch = epoch:maxepoch
    tic

    targetchunk = mod(epoch-1, numchunks)+1;
    
    [Wu, bu] = unpack(paramsp);


    y = cell(numchunks, numlayers+1);
    x = cell(numchunks, numlayers+1);
    
    if storeD
        dEdy = cell(numchunks, numlayers+1);
        dEdx = cell(numchunks, numlayers);
    end


    grad = mzeros(psize,1);
    grad2 = mzeros(psize,1);
    
    ll = 0;

    %forward prop:
    %index transition takes place at nonlinearity
    for chunk = 1:numchunks
        
        y{chunk, 1} = store(indata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        yip1 = conv( y{chunk, 1} );

        dEdW = cell(numlayers, 1);
        dEdb = cell(numlayers, 1);

        dEdW2 = cell(numlayers, 1);
        dEdb2 = cell(numlayers, 1);

        for i = 1:numlayers

            yi = yip1;
            yip1 = [];
            xi = Wu{i}*yi + repmat(bu{i}, [1 sizechunk]);
            yi = [];

            if strcmp(layertypes{i}, 'logistic')
                yip1 = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yip1 = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yip1 = xi;
            elseif strcmp( layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yip1 = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );
                tmp = [];
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            y{chunk, i+1} = store(yip1);
        end

        %back prop:
        %cross-entropy for logistics:
        %dEdy{numlayers+1} = outdata./y{numlayers+1} - (1-outdata)./(1-y{numlayers+1});
        %cross-entropy for softmax:
        %dEdy{numlayers+1} = outdata./y{numlayers+1};

        if hybridmode && chunk ~= targetchunk
            y{chunk, numlayers+1} = []; %save memory
        end

        outc = conv(outdata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        
        if rms || strcmp( layertypes{numlayers}, 'linear' )
            ll = ll + makeDouble( -sum(sum((outc - yip1).^2)) );
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                %ll = ll + makeDouble( sum(sum(outc.*log(yip1 + (outc==0)) + (1-outc).*log(1-yip1 + (outc==1)))) );
                %more stable:
                ll = ll + sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));                
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                ll = ll + makeDouble(sum(sum(outc.*log(yip1))));
            end
        end
        xi = [];
        
        
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    dEdxi = dEdyip1.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    dEdxi = dEdyip1;
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    dEdxi = outc - yip1; %simplified due to canonical link

                    %if strcmp(layertypes{i}, 'linear')
                    %    dEdxi = 2*dEdxi;  %the convention is to use the doubled version of the squared-error objective
                    %end

                    
                else
                    dEdyip1 = 2*(outc - yip1); %mult by 2 because we dont include the 1/2 before

                    if strcmp( layertypes{i}, 'softmax' )
                        dEdxi = dEdyip1.*yip1 - yip1.* repmat( sum( dEdyip1.*yip1, 1 ), [layersizes(i+1) 1] );
                        %error( 'RMS error not supported with softmax output' );

                    elseif strcmp(layertypes{i}, 'logistic')
                        dEdxi = dEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        dEdxi = dEdyip1;
                    else
                        error( 'Unknown/unsupported layer type' );
                    end

                    dEdyip1 = [];

                end

                outc = [];

            end
            dEdyi = Wu{i}'*dEdxi;

            if storeD && (~hybridmode || chunk == targetchunk)
                dEdx{chunk, i} = store(dEdxi);
                dEdy{chunk, i} = store(dEdyi);
            end

            yi = conv(y{chunk, i});

            if hybridmode && chunk ~= targetchunk
                y{chunk, i} = []; %save memory
            end

            %standard gradient comp:
            dEdW{i} = dEdxi*yi';
            dEdb{i} = sum(dEdxi,2);

            %gradient squared comp:
            dEdW2{i} = (dEdxi.^2)*(yi.^2)';
            dEdb2{i} = sum(dEdxi.^2,2);

            dEdxi = [];

            dEdyip1 = dEdyi;
            dEdyi = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];  dEdyip1 = [];

        if chunk == targetchunk
            gradchunk = pack(dEdW, dEdb);
            grad2chunk = pack(dEdW2, dEdb2);
        end

        grad = grad + pack(dEdW, dEdb);

        grad2 = grad2 + pack(dEdW2, dEdb2);

        %for checking F:
        %gradouter = gradouter + pack(dEdW, dEdb)*pack(dEdW, dEdb)';

        dEdW = []; dEdb = []; dEdW2 = []; dEdb2 = [];
    end
    
    grad = grad / conv(numcases);
    grad = grad - conv(weightcost)*(maskp.*paramsp);
    
    grad2 = grad2 / conv(numcases);
    
    gradchunk = gradchunk/conv(sizechunk) - conv(weightcost)*(maskp.*paramsp);
    grad2chunk = grad2chunk/conv(sizechunk);
    
    ll = ll / numcases;
    
    ll = ll - 0.5*weightcost*makeDouble(paramsp'*(maskp.*paramsp));
    
    
    oldll = ll;
    ll = [];
        
    %the following commented blocks of code are for checking the matrix
    %computation functions using finite differences.  If you ever add stuff
    %to the objective you should check that everything is correct using
    %methods like these (or something similar).  Be warned that if you use
    %hessiancsd (available online) you have to be mindful of what your
    %matrix-vector product implementation does if it's given complex values
    %in the input vector
    
    %for checking F:
    %gradouter = gradouter - grad*grad'*conv(1/numcases);
    %{
    lambda = 0.0;
    F = mzeros(psize);
    for j = 1:psize

        ej = mzeros(psize,1);
        ej(j) = 1;

        F(:,j) = computeFV(ej);
    end

    Fexact = gradouter;

    1==1;
    %}


    %H computation check
    %{
    if epoch == 1
        lambda = 0;

        Hfinite = hessiancsd(@(p)computeLL(p, indata, outdata), paramsp);
        %[f,g,Hfinite] = autoHess(paramsp, 0, @(p)computeLL(p, indata, outdata));
        %Hfinite = 0;

        Hexact = zeros(psize);
        for j = 1:psize
            ej = zeros(psize,1);
            ej(j) = 1;

            Hexact(:,j) = computeHV( ej );
        end

        1==1;
    end        
    %}

    %G computation check:
    %{
    estep = 1e-6;
    Gp = zeros(psize);
    dY = zeros(size(outdata,1), psize);
    for n = 1:numcases

        Pbase = computePred( paramsp, conv(indata(:,n)) );

        for j = 1:psize

            Wd = paramsp;
            Wd(j) = Wd(j) + estep;

            dY(:,j) = (computePred( Wd, conv(indata(:,n)) ) - Pbase)/estep;
        end

        %softmax:
        %Gp = Gp + dY'*(diag(y{numlayers+1}(:,n)) - y{numlayers+1}(:,n)*y{numlayers+1}(:,n)')*dY;

        %logistic:
        if ~rms
            %Gp = Gp + dY'*(-diag(  y{numlayers+1}(:,n).*(1-y{numlayers+1}(:,n))  ))*dY;
            Gp = Gp + -2*dY'*dY;
        else
            %{
            yip1 = y{numlayers+1}(:,n);

            dEdyip1 = 2*(outdata(:,n) - yip1); %mult by 2 because we dont include the 1/2 before
            dd = -2*yip1.*(1-yip1);
            dEdxi = dEdyip1.*yip1.*(1-yip1);

            Hm = diag(  dEdyip1.*yip1.*(1-yip1).*(1-2*yip1) + dd.*yip1.*(1-yip1)  );

            dEdyip1 = []; dd = [];
            %}

            yip1 = y{numlayers+1}(:,n);
            Hm = diag( -2* (yip1.*(1-yip1)).^2 );

            %Hm = -2;
            %Gp = Gp + dY'*Hm*dY;
        end

    end
    Gp = Gp / conv(numcases);
    
    lambda = 0.0;
    G = zeros(psize);
    for j = 1:psize

        Wd = zeros(psize,1);
        Wd(j) = 1;

        G(:,j) = computeGV(Wd);
    end
    1==1;    
    %}
      
    %maxiters is the most important variable that you should try
    %tweaking.  While the ICML paper had maxiters=250 for everything
    %I've since found out that this wasn't optimal.  For example, with
    %pre-trained weights for CURVES, maxiters=150 is better.  And for
    %the FACES dataset you should use something like maxiters=100.
    %Setting it too small or large can be bad to various degrees.
    %Currently I'm trying to automate"this choice, but it's quite hard
    %to come up with a *robust* heuristic for doing this.
    
    maxiters = 50;
    miniters = 1;
    outputString(['maxiters = ' num2str(maxiters) '; miniters = ' num2str(miniters)]);
    
    %preconditioning vector.  Feel free to experiment with this.  For
    %some problems (like the RNNs) this style of diaognal precondition
    %doesn't seem to be beneficial.  Probably because the parameters don't
    %exibit any obvious "axis-aligned" scaling issues like they do with
    %standard deep neural nets
    precon = (grad2 + mones(psize,1)*conv(lambda) + maskp*conv(weightcost)).^(3/4);
    %precon = mones(psize,1);   
    if lambda < thresh
        if strcmp(layertypes{numlayers}, 'logistic')
            [chs, iterses] = conjgrad_1(@(V)-computeBV(V), grad - 0.5 * (term1(delta) + term2(delta)), ch * conv(decay), ceil(maxiters), ceil(miniters), precon, jacket);            
        elseif strcmp(layertypes{numlayers}, 'linear')
            [chs, iterses] = conjgrad_1(@(V)-computeBV(V), grad - 0.5 * term1(delta), ch * conv(decay), ceil(maxiters), ceil(miniters), precon, jacket);                                  
        elseif strcmp(layertypes{numlayers}, 'softmax')
            [chs, iterses] = conjgrad_1(@(V)-computeBV(V), grad - 0.5 * (term1(delta) + term2(delta)), ch * conv(decay), ceil(maxiters), ceil(miniters), precon, jacket);
        end
    else        
        [chs, iterses] = conjgrad_1( @(V)-computeBV(V), grad, ch * conv(decay), ceil(maxiters), ceil(miniters), precon, jacket);
    end
    %slightly decay the previous change vector before using it as an
    %initialization.  This is something I didn't mention in the paper,
    %and it's not overly important but it can help a lot in some situations 
    %so you should probably use it

    ch = chs{end};
    
    iters = iterses(end);

    totalpasses = totalpasses + iters;
    outputString(['CG steps used: ' num2str(iters) ' Total is: ' num2str(totalpasses) ]);

    p = ch;
    
    j = length(chs);
    
    %"CG-backtracking":
    %It is not clear what subset of the data you should perform this on.
    %If possible you can use the full training set, as the uncommented block
    %below does.  Otherwise you could use some other set, like the current
    %mini-batch set, although that *could* be worse in some cases.  You can
    %also try not using it at all, or implementing it better so that it doesn't
    %require the extra storage

    %version with no backtracking:
    %{
    [ll, err] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);
    %}
    
    %current mini-batch version:
    %{
    [ll_chunk, err_chunk] = computeLL(paramsp + p, indata, outdata, numchunks, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, numchunks, targetchunk);

    for j = (length(chs)-1):-1:1
        [lowll_chunk, lowerr_chunk] = computeLL(paramsp + chs{j}, indata, outdata, numchunks, targetchunk);

        if ll_chunk > lowll_chunk
            j = j+1;
            break;
        end

        ll_chunk = lowll_chunk;
        err_chunk = lowerr_chunk;
    end
    if isempty(j)
        j = 1;
    end
    [ll, err] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);
    %}

    %full training set version:
    
    [ll, err] = computeLL(paramsp + p, indata, outdata, numchunks);
    for j = (length(chs)-1):-1:1
        [lowll, lowerr] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);

        if ll > lowll
            j = j+1;
            break;
        end

        ll = lowll;
        err = lowerr;
    end
    if isempty(j)
        j = 1;
    end
    
    p = chs{j};
    outputString( ['Chose iters : ' num2str(iterses(j))] );
    
    
    [ll_chunk, err_chunk] = computeLL(paramsp + chs{j}, indata, outdata, numchunks, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, numchunks, targetchunk);

    %disabling the damping when computing rho is something I'm not 100% sure
    %about.  It probably doesn't make a huge difference either way.  Also this
    %computation could probably be done on a different subset of the training data
    %or even the whole thing
    autodamp = 0;
    denom = -0.5*makeDouble(chs{j}'*computeBV(chs{j})) - makeDouble(grad'*chs{j});
    autodamp = 1;
    rho = (oldll_chunk - ll_chunk)/denom;
    if oldll_chunk - ll_chunk > 0
        rho = -Inf;
    end

    outputString( ['rho = ' num2str(rho)] );




    %bog-standard back-tracking line-search implementation:
    rate = 1.0;    

    c = 10^(-2);
    k = 0;
    while k < 60

        if ll >= oldll + c*makeDouble(grad'*conv(rate)*chs{j})
            break;
        else
            rate = 0.8*rate;
            k = k + 1;
            %outputString('#');
        end

        %this is computed on the whole dataset.  If this is not possible you can
        %use another set such the test set or a seperate validation set
        [ll, err] = computeLL(paramsp + conv(rate)*chs{j}, indata, outdata, numchunks);
    end

    if k == 60
        %completely reject the step
        k = Inf;
        rate = 0.0;
        ll = oldll;
    end

    outputString( ['Number of reductions : ' num2str(k) ', chosen rate: ' num2str(rate)] );

    %the damping heuristic (also very standard in optimization):
    outputString( ['Old lambda: ' num2str(lambda)]);
    if autodamp
        if rho < 0.25 || isnan(rho)
            lambda = lambda*boost;
        elseif rho > 0.75
            lambda = lambda*drop;
        end
        outputString(['New lambda: ' num2str(lambda)]);
    end
        
    %Parameter update:   
    delta = conv(rate) * p;    
    chs = [];
    outputString( ['delta norm : ' num2str(norm(delta))]);
    
    %{
    %Random perturbation to detect canyons.
    
    [old_ll_c, old_err_c] = computeLL(paramsp, indata, outdata, numchunks);
    outputString(['old_ll: ' num2str(old_ll_c) ' old_err: ' num2str(old_err_c)])
    for k = 1:20
        test = randn(size(delta));
        test = test / norm(test) * norm(delta) / 2;
        [ll_c, err_c] = computeLL(paramsp + test, indata, outdata, numchunks);
        if ll_c < old_ll_c
            outputString(['test: ' num2str(k) ' ll: ' num2str(ll_c) ' err: ' num2str(err_c) ' + ']);
        else
            outputString(['test: ' num2str(k) ' ll: ' num2str(ll_c) ' err: ' num2str(err_c) ' - ']);
        end    
    end    
    %}
    if epoch == 1
        [init_ll, init_err] = computeLL(paramsp, indata, outdata, numchunks);
        [init_test_ll, init_test_err] = computeLL(paramsp, intest, outtest, numchunks_test);
        outputString(['epoch: 0, Log likelihood: ' num2str(init_ll) ', error rate: ' num2str(init_err) ]);
        outputString(['TEST Log likelihood: ' num2str(init_test_ll) ', error rate: ' num2str(init_test_err) ]);
    end    

    paramsp = paramsp + delta;
    
    lambdarecord(epoch,1) = lambda;

    llrecord(epoch,1) = ll;
    errrecord(epoch,1) = err;    
    times(epoch) = toc;
    outputString( ['epoch: ' num2str(epoch) ', Log likelihood: ' num2str(ll) ', error rate: ' num2str(err) ] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest, numchunks_test);
    llrecord(epoch,2) = ll_test;
    errrecord(epoch,2) = err_test;
    outputString( ['TEST Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test) ] );
    
    outputString( ['Error rate difference (test - train): ' num2str(err_test-err)] );
    outputString( ['Time elapsed : ' num2str(toc)] );
    outputString( '' );

    pause(0)
    drawnow
    
    tmp = paramsp;
    paramsp = single(paramsp);    
    tmp3 = ch;
    ch = single(ch);    
    save( [runName '_nnet_running.mat'], 'paramsp', 'ch', 'epoch', 'lambda', 'totalpasses', 'llrecord', 'times', 'errrecord', 'lambdarecord' );
    if mod(epoch,100) == 0
        save( [runName '_nnet_epoch' num2str(epoch) '.mat'], 'paramsp', 'ch', 'epoch', 'lambda', 'totalpasses', 'llrecord', 'times', 'errrecord', 'lambdarecord' );
    end
    paramsp = tmp;    
    ch = tmp3;    

    clear tmp tmp3

end

paramsp = makeDouble(paramsp);

outputString( ['Total time: ' num2str(sum(times)) ] );

fclose(fid);

end
