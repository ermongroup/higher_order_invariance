function nnet_experiments(dataset, algorithm, runName)
%{
    dataset: can be 'CURVES', 'MNIST', 'FACES', or 'MNIST_classification'
    algorithm: can be 'ng', 'geo', 'mid', 'geo_faster' or 'adam'
    runName: the name of logs
%}

    if strcmp(dataset, 'MNIST') == 1 || strcmp(dataset, 'MNIST_classification') == 1
        thresh = 10;
    elseif strcmp(dataset, 'CURVES') == 1
        thresh = 5;
    elseif strcmp(dataset, 'FACES') == 1
        thresh = 0.1;
    end            
    jacket = 1;
    maxepoch = 140;

    seed = 1234;

    randn('state', seed );
    rand('twister', seed+1 );

    %you will NEVER need more than a few hundred epochs unless you are doing
    %something very wrong.  Here 'epoch' means parameter update, not 'pass over
    %the training set'.
    %uncomment the appropriate section to use a particular dataset

    if strcmp(dataset, 'MNIST')
        %%%%%%%%
        % MNIST
        %%%%%%%%        
        %dataset available at www.cs.toronto.edu/~jmartens/mnist_all.mat

        load mnist_all
        traindata = zeros(0, 28^2);
        for i = 0:9
            eval(['traindata = [traindata; train' num2str(i) '];']);
        end
        %indata = double(traindata)/255;
        indata = single(traindata)/255;
        clear traindata

        testdata = zeros(0, 28^2);
        for i = 0:9
            eval(['testdata = [testdata; test' num2str(i) '];']);
        end
        %intest = double(testdata)/255;
        intest = single(testdata)/255;
        clear testdata

        indata = indata';
        intest = intest';

        perm = randperm(size(intest,2));
        intest = intest( :, perm );

        randn('state', seed );
        rand('twister', seed+1 );

        perm = randperm(size(indata,2));
        indata = indata( :, perm );

        layersizes = [1000 500 250 30 250 500 1000];
        layertypes = {'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic'};

        %standard L_2 weight-decay:
        weightcost = 1e-5;

        numchunks = 8;
        numchunks_test = 8;
        %%%%%%%%
        %it's an auto-encoder so output is input
        %reduce dimension
        outdata = indata;
        outtest = intest;
        errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

    elseif strcmp(dataset, 'MNIST_classification')    
        %%%%%%%%
        % MNIST_classification
        %%%%%%%%        
        load mnist_all
        traindata = zeros(0, 28^2);    
        trainlabels = zeros(0, 10);
        for i = 0:9
            eval(['traindata = [traindata; train' num2str(i) '];']);
            this = eval(['train' num2str(i)]);
            onehot = zeros(10,size(this, 1));
            onehot(i + 1, :) = 1;
            trainlabels = [trainlabels onehot];
        end
        %indata = double(traindata)/255;
        indata = single(traindata)/255;    
        clear traindata

        testdata = zeros(0, 28^2);
        testlabels = zeros(10, 0);
        for i = 0:9
            eval(['testdata = [testdata; test' num2str(i) '];']);
            this = eval(['test' num2str(i)]);
            onehot = zeros(10,size(this,1));
            onehot(i + 1, :) = 1;
            testlabels = [testlabels onehot];
        end
        %intest = double(testdata)/255;
        intest = single(testdata)/255;
        clear testdata
        alldata = [indata; intest];
        meanvalue = mean(alldata(:));
        stdvalue = std(alldata(:));
        indata = (indata - meanvalue) ./ stdvalue;
        indata = indata';
        outdata = trainlabels;
        intest = (intest - meanvalue) ./ stdvalue;
        intest = intest';
        outtest = testlabels;

        perm = randperm(size(intest,2));
        intest = intest( :, perm );
        outtest = outtest(:, perm);

        randn('state', seed );
        rand('twister', seed+1 );

        perm = randperm(size(indata,2));
        indata = indata( :, perm );
        outdata = outdata(:, perm );

        layersizes = [1000 500 250 30];
        layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'softmax'};

        %standard L_2 weight-decay:
        weightcost = 1e-5;

        numchunks = 8;
        numchunks_test = 8;
        errtype = 'class';

    elseif strcmp(dataset, 'FACES')
        %%%%%%%%
        % FACES
        %%%%%%%%        
        %dataset available at www.cs.toronto.edu/~jmartens/newfaces_rot_single.mat
        load newfaces_rot_single
        total = 165600;
        trainsize = (total/40)*25;
        testsize = (total/40)*10;
        indata = newfaces_single(:, 1:trainsize);
        intest = newfaces_single(:, (end-testsize+1):end);
        clear newfaces_single


        perm = randperm(size(intest,2));
        intest = intest( :, perm );
        %randn('state', seed );
        %rand('twister', seed+1 );

        perm = randperm(size(indata,2));
        %disp('Using 1/2');
        %perm = perm( 1:size(indata,1)/2 );
        indata = indata( :, perm );
        %outdata = outdata( :, perm );

        layertypes = {'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'linear'};
        layersizes = [2000 1000 500 30 500 1000 2000];

        %standard L_2 weight-decay:
        weightcost = 1e-5;
        weightcost = weightcost / 2; %an older version of the code used in the paper had a differently scaled objective (by 2) in the case of linear output units.  Thus we now need to reduce weightcost by a factor 2 to be consistent

        numchunks = 20;
        numchunks_test = 8;
        %%%%%%%%
        %it's an auto-encoder so output is input
        %reduce dimension
        outdata = indata;
        outtest = intest;
        errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)
    elseif strcmp(dataset, 'CURVES')
        %%%%%%%%
        % CURVES
        %%%%%%%%        
        %dataset available at www.cs.toronto.edu/~jmartens/digs3pts_1.mat
        tmp = load('digs3pts_1.mat');
        indata = tmp.bdata';
        %outdata = tmp.bdata;
        intest = tmp.bdatatest';
        %outtest = tmp.bdatatest;
        clear tmp

        perm = randperm(size(indata,2));
        %disp('Using 1/2');
        %perm = perm( 1:size(indata,1)/2 );
        indata = indata( :, perm );
        %outdata = outdata( :, perm );

        layersizes = [400 200 100 50 25 6 25 50 100 200 400];
        layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic'};

        %standard L_2 weight-decay:
        weightcost = 1e-5;
        numchunks = 4;
        numchunks_test = 4;
        %%%%%%%%

        %it's an auto-encoder so output is input
        %reduce dimension
        outdata = indata;
        outtest = intest;
        errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)
    end



    runDesc = ['seed = ' num2str(seed) ', enter anything else you want to remember here' ];

    %next try using autodamp = 0 for rho computation.  both for version 6 and
    %versions with rho and cg-backtrack computed on the training set

    resumeFile = [];

    paramsp = [];
    Win = [];
    bin = [];
    %[Win, bin] = loadPretrainedNet_curves;

    mattype = 'gn'; %Gauss-Newton.  The other choices probably won't work for whatever you're doing
    %mattype = 'hess';
    %mattype = 'empfish';

    rms = 0;

    hybridmode = 1;

    %decay = 1.0;
    decay = 0.95;

    if strcmp(algorithm, 'ng')
        nnet_train_ng( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
    elseif strcmp(algorithm, 'geo')
        nnet_train_geo(thresh, runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
    elseif strcmp(algorithm, 'geo_faster')
        nnet_train_geo_faster(thresh, runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
    elseif strcmp(algorithm, 'midpoint')
        nnet_train_midpoint_2(thresh, runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);        
    elseif strcmp(algorithm, 'adam')
        nnet_train_adam( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
    end
end
