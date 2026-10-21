function TurboNet(steps)

if nargin < 1, steps = 1:7; end
if ischar(steps) && strcmpi(steps, 'params')
    for v = {'lean', 'lean_noD', 'lean_noMS', 'full'}
        fprintf('1D %-14s %8d parameters\n', v{1}, tn_nparams(tn_buildTurboNet1D(10000, 4, v{1})));
    end
    for v = {'lean_noMS', 'lean_noMS_noD', 'lean', 'full'}
        fprintf('2D %-14s %8d parameters\n', v{1}, tn_nparams(tn_buildTurboNet2D([128 128 4], 4, v{1})));
    end
    return
end
cfg = tn_config();
fprintf('Turbo_Net | data: %s | results: %s\n', cfg.csvRoot, cfg.resDir);
diary(fullfile(cfg.resDir, sprintf('log_%s.txt', datestr(now, 'yyyymmdd_HHMM'))));
cleanup = onCleanup(@() diary('off'));

if ismember(1, steps), step1_csv2mat(cfg);        end
if ismember(2, steps), step2_make_dataset(cfg);   end
if ismember(3, steps), step3_train_1D(cfg);       end
if ismember(4, steps), step4_train_2D(cfg);       end
if ismember(5, steps), step5_noise_test(cfg);     end
if ismember(6, steps), step6_ablation(cfg);       end
if ismember(7, steps)
    try, reset(gpuDevice); catch, end
    close all force; drawnow;
    step7_tables_figures(cfg);
end
end

function cfg = tn_config()
cfg.codeDir = fileparts(mfilename('fullpath'));

cfg.csvRoot = fullfile(cfg.codeDir, 'DATA CSV');
if ~exist(cfg.csvRoot, 'dir') && exist(fullfile(cfg.codeDir, '..', 'DATA CSV'), 'dir')
    cfg.csvRoot = fullfile(cfg.codeDir, '..', 'DATA CSV');
end
if ~exist(cfg.csvRoot, 'dir') && exist(fullfile(cfg.codeDir, '..', 'THUC NGHIEM_0808', 'Data', 'DATA CSV'), 'dir')
    cfg.csvRoot = fullfile(cfg.codeDir, '..', 'THUC NGHIEM_0808', 'Data', 'DATA CSV');
end
cfg.workDir = fullfile(cfg.codeDir, 'Work');
cfg.matDir  = fullfile(cfg.workDir, 'mat');
cfg.resDir  = fullfile(cfg.codeDir, 'Result');
if ~exist(cfg.resDir, 'dir') && exist(fullfile(cfg.codeDir, '..', 'Result'), 'dir')
    cfg.resDir = fullfile(cfg.codeDir, '..', 'Result');
end

cfg.sessions  = {'S1','S2','S3','S4','S5','S6','S7'};
cfg.levels    = {'P0','P1','P2','P3'};
cfg.pressure  = [4 3 2 1.5];
cfg.speeds    = [1500 2000 2500];
cfg.chanNames = {'A','B','C','D'};

cfg.decim      = 2;
cfg.winLen     = 10000;
cfg.nWinPerRec = 200;
cfg.stft = struct('win', 320, 'hop', 76, 'nfft', 320, 'nFreq', 128, 'nTime', 128);

cfg.split = [0.70 0.20 0.10];
cfg.nRuns = 10;
cfg.seed0 = 2026;

cfg.train.miniBatch       = 64;
cfg.train.maxEpochs       = 30;
cfg.train.patienceScratch = Inf;
cfg.train.patience        = 8;
cfg.train.lrScratch       = 1e-3;
cfg.train.lrPretrain      = 1e-4;
cfg.train.epochsPre       = 20;
cfg.train.l2              = 1e-4;
cfg.train.dropPeriod      = 15;
cfg.train.dropFactor      = 0.2;
cfg.train.verbose         = true;
cfg.train.exec            = 'auto';

cfg.svm.boxGrid = [0.1 1 10 100];
cfg.rf.nTrees   = 300;
cfg.rf.leafGrid = [1 5];

cfg.noiseSNR = [20 10 5 0];
cfg.nRunsAbl = 3;

cfg.mainVariant1D = 'lean';
cfg.mainVariant2D = 'lean_noMS';
cfg.ablVariants1D = {'lean', 'lean_noD', 'lean_noMS', 'full'};
cfg.ablVariants2D = {'lean_noMS', 'lean_noMS_noD', 'lean', 'full'};

cfg.maxRows = Inf;
cfg.quick   = strcmp(getenv('TURBONET_QUICK'), '1');
if cfg.quick
    cfg.maxRows = 1e6;  cfg.nWinPerRec = 20;  cfg.nRuns = 1;  cfg.nRunsAbl = 1;
    cfg.train.maxEpochs = 2;  cfg.train.epochsPre = 1;  cfg.train.miniBatch = 16;
    cfg.workDir = fullfile(cfg.codeDir, 'Work_quick');
    cfg.matDir  = fullfile(cfg.workDir, 'mat');
    cfg.resDir  = fullfile(cfg.codeDir, 'Result_quick');
end
cfg.figDir = fullfile(cfg.resDir, 'Figures');
cfg.netDir = fullfile(cfg.resDir, 'Nets');
for d = {cfg.workDir, cfg.matDir, cfg.resDir, cfg.figDir, cfg.netDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end
end

function step1_csv2mat(cfg)
fprintf('\n=== STEP 1: CSV -> MAT (decimate x%d) ===\n', cfg.decim);
if ~exist(cfg.csvRoot, 'dir'), error('Data folder not found. Edit cfg.csvRoot in tn_config.'); end
rows = cell(0, 14);
for is = 1:numel(cfg.sessions)
    for ip = 1:numel(cfg.levels)
        for iv = 1:numel(cfg.speeds)
            S = cfg.sessions{is};  P = cfg.levels{ip};  v = cfg.speeds(iv);
            out = fullfile(cfg.matDir, sprintf('%s_%s_%d.mat', S, P, v));
            if exist(out, 'file')
                q = load(out, 'info');  info = q.info;
            else
                t0 = tic;
                [X0, fs0, info] = tn_readPicoCSV(fullfile(cfg.csvRoot, S, P, sprintf('%d.csv', v)), cfg.maxRows);
                X = zeros(ceil(size(X0, 1) / cfg.decim), 4, 'single');
                for c = 1:4, X(:, c) = single(decimate(X0(:, c), cfg.decim)); end
                fs = fs0 / cfg.decim;
                Xc = X0 - mean(X0, 1);
                info.rmsV = sqrt(mean(Xc.^2, 1));
                info.peakV = max(abs(Xc), [], 1);
                info.clipFrac = mean(abs(X0) >= 0.99, 1);
                info.fs0 = fs0;  info.fs = fs;
                save(out, 'X', 'fs', 'info', '-v7');
                fprintf('  %s_%s_%d: %d samples, %.2f s (%.1f s)\n', S, P, v, info.nRows, info.duration, toc(t0));
            end
            rows(end+1, :) = {S, P, cfg.pressure(ip), v, info.nRows, info.duration, info.fs0, info.nNaN, ...
                info.rmsV(1), info.rmsV(2), info.rmsV(3), info.rmsV(4), max(info.peakV), max(info.clipFrac)};
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Session','Level','Pressure_kgcm2','Speed_rpm','nSamples', ...
    'Duration_s','Fs_Hz','nOverrange','RMS_A_V','RMS_B_V','RMS_C_V','RMS_D_V','MaxPeak_V','MaxClipFrac'});
writetable(T, fullfile(cfg.resDir, 'QC_records.csv'));
end

function [X, fs0, info] = tn_readPicoCSV(f, maxRows)
fid = fopen(f, 'r', 'n', 'UTF-8');
if fid < 0, error('Cannot open %s', f); end
closer = onCleanup(@() fclose(fid));
opt = {'Delimiter', ',', 'HeaderLines', 3, 'CollectOutput', true, ...
       'TreatAsEmpty', {char(8734), ['-' char(8734)], ['+' char(8734)]}, 'EmptyValue', NaN};
if isinf(maxRows)
    C = textscan(fid, '%f%f%f%f%f', opt{:});
    if ~feof(fid), C = {readmatrix(f, 'NumHeaderLines', 3)}; end
else
    C = textscan(fid, '%f%f%f%f%f', maxRows, opt{:});
end
M = C{1};
t = M(:, 1);  X = M(:, 2:5);
fs0 = 1 / median(diff(t(1:min(end, 1e5))));
info.nRows = size(X, 1);
info.duration = t(end) - t(1);
info.nNaN = sum(isnan(X(:)));
if info.nNaN > 0, X = fillmissing(fillmissing(X, 'previous'), 'next'); end
end

function step2_make_dataset(cfg)
fprintf('\n=== STEP 2: windows / features / spectrograms ===\n');
fW = fullfile(cfg.workDir, 'dataset_windows.mat');
fF = fullfile(cfg.workDir, 'dataset_features.mat');
fS = fullfile(cfg.workDir, 'dataset_spec.mat');
if exist(fW, 'file') && exist(fF, 'file') && exist(fS, 'file'), fprintf('  [exists]\n'); return; end

L = cfg.winLen;  W = cfg.nWinPerRec;
N = numel(cfg.sessions) * numel(cfg.levels) * numel(cfg.speeds) * W;
X = zeros(L, 4, N, 'single');
recId = zeros(N,1); sess = strings(N,1); lev = strings(N,1); pres = zeros(N,1); spd = zeros(N,1); win = zeros(N,1); t0 = zeros(N,1);
r = 0;  n = 0;
for is = 1:numel(cfg.sessions)
    for ip = 1:numel(cfg.levels)
        for iv = 1:numel(cfg.speeds)
            r = r + 1;
            S = cfg.sessions{is};  P = cfg.levels{ip};  v = cfg.speeds(iv);
            q = load(fullfile(cfg.matDir, sprintf('%s_%s_%d.mat', S, P, v)), 'X', 'fs');
            fs = q.fs;  nS = size(q.X, 1);
            if nS < W * L, error('Record %s_%s_%d too short', S, P, v); end
            starts = round(linspace(1, nS - L + 1, W));
            for w = 1:W
                n = n + 1;
                seg = q.X(starts(w):starts(w)+L-1, :);
                X(:, :, n) = seg - mean(seg, 1);
                recId(n) = r; sess(n) = S; lev(n) = P; pres(n) = cfg.pressure(ip);
                spd(n) = v; win(n) = w; t0(n) = (starts(w) - 1) / fs;
            end
        end
    end
end
meta = table(recId, categorical(sess), categorical(lev, cfg.levels), pres, spd, win, t0, ...
    'VariableNames', {'recId','session','label','pressure','speed','win','t0'});
save(fW, 'X', 'meta', 'fs', '-v7.3', '-nocompression');
fprintf('  %d windows (%d per class), fs = %.1f Hz\n', N, N/numel(cfg.levels), fs);

tt = tic;  [F, featNames] = tn_features(X, fs);  featMs = 1000 * toc(tt) / N;
save(fF, 'F', 'featNames', 'featMs');

rng(cfg.seed0);
Ssub = tn_spectrogram(X(:, :, randperm(N, min(N, 2000))), cfg);
lo = zeros(1, 4);  hi = zeros(1, 4);
for c = 1:4
    v = Ssub(:, :, c, :);
    lo(c) = prctile(v(:), 0.5);  hi(c) = prctile(v(:), 99.95);
end
clear Ssub v
U = zeros(cfg.stft.nFreq, cfg.stft.nTime, 4, N, 'uint8');
tt = tic;
for n0 = 1:1000:N
    nn = n0:min(N, n0 + 999);
    U(:, :, :, nn) = tn_quantize(tn_spectrogram(X(:, :, nn), cfg), lo, hi);
end
specMs = 1000 * toc(tt) / N;
fAxis = (0:cfg.stft.nFreq-1) * fs / cfg.stft.nfft;
tAxis = ((0:cfg.stft.nTime-1) * cfg.stft.hop + cfg.stft.win/2) / fs;
save(fS, 'U', 'lo', 'hi', 'fAxis', 'tAxis', 'specMs', '-v7.3', '-nocompression');
end

function [F, names] = tn_features(X, fs)
[L, C, N] = size(X);
nb = 16;  edges = logspace(log10(100), log10(40000), nb + 1);
f = (0:floor(L/2))' * fs / L;
band = f >= 50 & f <= 40000;  fb = f(band);
bandIdx = arrayfun(@(b) f >= edges(b) & f < edges(b+1), 1:nb, 'UniformOutput', false);
win = hann(L);  nPer = 30;
F = zeros(N, nPer*C + 3);  rmsAll = zeros(N, C);
for n0 = 1:500:N
    nn = n0:min(N, n0 + 499);
    for c = 1:C
        x  = double(reshape(X(:, c, nn), L, []));
        x  = x - mean(x, 1);
        r  = sqrt(mean(x.^2, 1));  sd = std(x, 0, 1);  pk = max(abs(x), [], 1);
        pp = max(x, [], 1) - min(x, [], 1);  ma = mean(abs(x), 1);  m2 = mean(x.^2, 1);
        T  = [r; sd; pk; pp; pk./r; mean(x.^4, 1)./m2.^2; mean(x.^3, 1)./m2.^1.5; ...
              r./ma; pk./ma; pk./mean(sqrt(abs(x)), 1).^2];
        P  = abs(fft(x .* win)).^2;  P = P(1:floor(L/2)+1, :);
        E  = zeros(nb, numel(nn));
        for b = 1:nb, E(b, :) = log10(sum(P(bandIdx{b}, :), 1) + eps); end
        Pb = P(band, :);  Pn = Pb ./ sum(Pb, 1);
        cen = sum(fb .* Pn, 1);  spr = sqrt(sum((fb - cen).^2 .* Pn, 1));
        ent = -sum(Pn .* log2(Pn + eps), 1) / log2(numel(fb));
        flt = exp(mean(log(Pb + eps), 1)) ./ mean(Pb, 1);
        F(nn, (c-1)*nPer + (1:nPer)) = [T; E; cen/1000; spr/1000; ent; flt]';
        rmsAll(nn, c) = r';
    end
end
F(:, end-2:end) = log(rmsAll(:, 1:3) ./ rmsAll(:, 4));
base = [{'RMS','STD','Peak','P2P','Crest','Kurt','Skew','Shape','Impulse','Clearance'}, ...
        arrayfun(@(b) sprintf('E%02d', b), 1:nb, 'UniformOutput', false), ...
        {'SpCentroid','SpSpread','SpEntropy','SpFlatness'}];
names = {};
ch = 'ABCD';
for c = 1:C, names = [names, strcat(ch(c), '_', base)]; end
names = [names, {'logRMS_A_D','logRMS_B_D','logRMS_C_D'}];
end

function S = tn_spectrogram(X, cfg)
[~, C, N] = size(X);
w = cfg.stft.win;  h = cfg.stft.hop;  nF = cfg.stft.nFreq;  nT = cfg.stft.nTime;
idx = (1:w)' + (0:nT-1) * h;
win = single(hann(w, 'periodic'));
S = zeros(nF, nT, C, N, 'single');
for n0 = 1:512:N
    nn = n0:min(N, n0 + 511);  nb = numel(nn);
    for c = 1:C
        x  = reshape(X(:, c, nn), [], nb);  x = x - mean(x, 1);
        Fr = fft(reshape(x(idx(:), :), w, nT, nb) .* win, cfg.stft.nfft, 1);
        S(:, :, c, nn) = reshape(10*log10(abs(Fr(1:nF, :, :)).^2 + 1e-12), nF, nT, 1, nb);
    end
end
end

function U = tn_quantize(S, lo, hi)
C = size(S, 3);
U = uint8(255 * min(max((S - reshape(lo, 1, 1, C)) ./ (reshape(hi, 1, 1, C) - reshape(lo, 1, 1, C)), 0), 1));
end

function step3_train_1D(cfg)
fprintf('\n=== STEP 3: 1D group (Turbo_Net-1D, SVM, RF) ===\n');
D  = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'X', 'meta', 'fs');
Fq = load(fullfile(cfg.workDir, 'dataset_features.mat'));
Y  = D.meta.label;
classes = categorical(cfg.levels, cfg.levels);
hasStats = ~isempty(which('fitcecoc')) && ~isempty(which('TreeBagger'));
if ~hasStats, warning('Statistics and Machine Learning Toolbox not found: SVM and RF are skipped.'); end
svmMdl = [];  rfMdl = [];
for run = 1:cfg.nRuns
    out = fullfile(cfg.resDir, sprintf('R1D_run%02d.mat', run));
    if exist(out, 'file'), fprintf('  [exists] %s\n', out); continue; end
    fprintf('\n--- Run %d/%d ---\n', run, cfg.nRuns);
    [itr, iva, ite] = tn_split(D.meta, cfg, run);
    res = [];
    muF = mean(Fq.F(itr, :), 1);  sgF = std(Fq.F(itr, :), 0, 1);  sgF(sgF == 0) = 1;
    Z = (Fq.F - muF) ./ sgF;
    if hasStats
        best = -inf;
        for C = cfg.svm.boxGrid
            rng(cfg.seed0 + run);  t = tic;
            mdl = fitcecoc(Z(itr, :), Y(itr), 'Coding', 'onevsone', 'Learners', ...
                  templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', C));
            tt = toc(t);  a = mean(predict(mdl, Z(iva, :)) == Y(iva));
            if a > best, best = a; svmMdl = mdl; tSvm = tt; bestC = C; end
        end
        t = tic;  yp = predict(svmMdl, Z(ite, :));  ti = toc(t);
        M = tn_metrics(Y(ite), yp, classes);
        res = [res, tn_pack('SVM', M, tSvm, 1000*ti/numel(ite) + Fq.featMs, NaN, yp, [], struct('C', bestC))];
        fprintf('  SVM          test acc = %.4f\n', M.acc);
        best = -inf;
        for leaf = cfg.rf.leafGrid
            rng(cfg.seed0 + run);  t = tic;
            mdl = TreeBagger(cfg.rf.nTrees, Z(itr, :), Y(itr), 'Method', 'classification', ...
                  'MinLeafSize', leaf, 'NumPredictorsToSample', round(sqrt(size(Z, 2))));
            tt = toc(t);  a = mean(categorical(predict(mdl, Z(iva, :)), cfg.levels) == Y(iva));
            if a > best, best = a; rfMdl = compact(mdl); tRf = tt; bestLeaf = leaf; end
        end
        t = tic;  yp = categorical(predict(rfMdl, Z(ite, :)), cfg.levels);  ti = toc(t);
        M = tn_metrics(Y(ite), yp, classes);
        res = [res, tn_pack('RF', M, tRf, 1000*ti/numel(ite) + Fq.featMs, NaN, yp, [], struct('leaf', bestLeaf))];
        fprintf('  RF           test acc = %.4f\n', M.acc);
    end
    [rT, net] = tn_trainTurbo1D(D.X, Y, itr, iva, ite, cfg, cfg.mainVariant1D, run);
    res = [res, rT];
    fprintf('  Turbo_Net-1D test acc = %.4f\n', rT.M.acc);
    yTrue = Y(ite);
    save(out, 'res', 'itr', 'iva', 'ite', 'yTrue');
    if run == 1
        netT1D = net;
        save(fullfile(cfg.netDir, 'run01_1D.mat'), 'netT1D', 'svmMdl', 'rfMdl', 'muF', 'sgF', '-v7.3');
    end
end
end

function step4_train_2D(cfg)
fprintf('\n=== STEP 4: 2D group (Turbo_Net-2D, GoogLeNet, AlexNet, SqueezeNet) ===\n');
Dm = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'meta');
Sp = load(fullfile(cfg.workDir, 'dataset_spec.mat'), 'U', 'specMs');
Y  = Dm.meta.label;
classes = categorical(cfg.levels, cfg.levels);
pre = {'googlenet', 'GoogLeNet'; 'alexnet', 'AlexNet'; 'squeezenet', 'SqueezeNet'};
for run = 1:cfg.nRuns
    out = fullfile(cfg.resDir, sprintf('R2D_run%02d.mat', run));
    if exist(out, 'file'), fprintf('  [exists] %s\n', out); continue; end
    fprintf('\n--- Run %d/%d ---\n', run, cfg.nRuns);
    [itr, iva, ite] = tn_split(Dm.meta, cfg, run);
    [rT, net] = tn_trainTurbo2D(Sp.U, Y, itr, iva, ite, cfg, cfg.mainVariant2D, run, Sp.specMs);
    res = rT;  nets = struct('netT2D', net);
    fprintf('  Turbo_Net-2D test acc = %.4f\n', rT.M.acc);
    for m = 1:size(pre, 1)
        net = imagePretrainedNetwork(pre{m, 1}, 'NumClasses', numel(cfg.levels));
        isz = net.Layers(1).InputSize;
        adsTr = augmentedImageDatastore(isz(1:2), Sp.U(:, :, 1:3, itr), Y(itr));
        adsVa = augmentedImageDatastore(isz(1:2), Sp.U(:, :, 1:3, iva), Y(iva));
        adsTe = augmentedImageDatastore(isz(1:2), Sp.U(:, :, 1:3, ite));  adsTe.MiniBatchSize = 128;
        opts = tn_trainOpts(cfg, adsVa, cfg.train.lrPretrain, cfg.train.epochsPre, numel(itr));
        rng(cfg.seed0 + 300 + 10*m + run);
        t = tic;  [net, info] = trainnet(adsTr, net, 'crossentropy', opts);  tt = toc(t);
        t = tic;  sc = minibatchpredict(net, adsTe);  ti = toc(t);
        yp = scores2label(sc, cfg.levels);
        M = tn_metrics(Y(ite), yp, classes);
        hist = struct('train', info.TrainingHistory, 'val', info.ValidationHistory);
        res = [res, tn_pack(pre{m, 2}, M, tt, 1000*ti/numel(ite) + Sp.specMs, tn_nparams(net), yp, hist)];
        nets.(['net' pre{m, 2}]) = net;
        fprintf('  %-12s test acc = %.4f\n', pre{m, 2}, M.acc);
    end
    yTrue = Y(ite);
    save(out, 'res', 'itr', 'iva', 'ite', 'yTrue');
    if run == 1, save(fullfile(cfg.netDir, 'run01_2D.mat'), '-struct', 'nets', '-v7.3'); end
end
end

function step5_noise_test(cfg)
fprintf('\n=== STEP 5: noise robustness ===\n');
D  = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'X', 'meta', 'fs');
Sp = load(fullfile(cfg.workDir, 'dataset_spec.mat'), 'lo', 'hi');
N1 = load(fullfile(cfg.netDir, 'run01_1D.mat'));
N2 = load(fullfile(cfg.netDir, 'run01_2D.mat'));
classes = categorical(cfg.levels, cfg.levels);
[~, ~, ite] = tn_split(D.meta, cfg, 1);
Xte = D.X(:, :, ite);  yt = D.meta.label(ite);  fs = D.fs;
clear D
pre = {'GoogLeNet', 'AlexNet', 'SqueezeNet'};
rows = {};
add = @(rows, snr, name, yp) [rows; {snr, name, tn_metrics(yt, yp, classes)}];
for snr = [Inf cfg.noiseSNR]
    rng(cfg.seed0 + 7);
    Xn = tn_addnoise(Xte, snr);
    if ~isempty(N1.svmMdl)
        Z = (tn_features(Xn, fs) - N1.muF) ./ N1.sgF;
        rows = add(rows, snr, 'SVM', predict(N1.svmMdl, Z));
        rows = add(rows, snr, 'RF', categorical(predict(N1.rfMdl, Z), cfg.levels));
    end
    sc = minibatchpredict(N1.netT1D, Xn, 'InputDataFormats', "TCB", 'MiniBatchSize', 256);
    rows = add(rows, snr, 'Turbo_Net-1D', scores2label(sc, cfg.levels));
    U  = tn_quantize(tn_spectrogram(Xn, cfg), Sp.lo, Sp.hi);
    sc = minibatchpredict(N2.netT2D, U, 'MiniBatchSize', 256);
    rows = add(rows, snr, 'Turbo_Net-2D', scores2label(sc, cfg.levels));
    for m = 1:numel(pre)
        net = N2.(['net' pre{m}]);  isz = net.Layers(1).InputSize;
        ads = augmentedImageDatastore(isz(1:2), U(:, :, 1:3, :));  ads.MiniBatchSize = 128;
        rows = add(rows, snr, pre{m}, scores2label(minibatchpredict(net, ads), cfg.levels));
    end
    fprintf('  SNR = %g dB done\n', snr);
end
T = table(cell2mat(rows(:, 1)), rows(:, 2), cellfun(@(M) M.acc, rows(:, 3)), ...
          cellfun(@(M) M.f1, rows(:, 3)), cellfun(@(M) M.detAcc, rows(:, 3)), ...
          'VariableNames', {'SNR_dB', 'Model', 'Accuracy', 'MacroF1', 'DetAcc'});
writetable(T, fullfile(cfg.resDir, 'Noise_robustness.csv'));
end

function Xn = tn_addnoise(X, snrDb)
if isinf(snrDb), Xn = X; return; end
Xn = X + sqrt(mean(X.^2, 1) / 10^(snrDb/10)) .* randn(size(X), 'like', X);
end

function step6_ablation(cfg)
fprintf('\n=== STEP 6: ablation ===\n');
D  = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'X', 'meta');
Sp = load(fullfile(cfg.workDir, 'dataset_spec.mat'), 'U');
Y  = D.meta.label;
V = {cfg.ablVariants1D, cfg.ablVariants2D};
mainV = {cfg.mainVariant1D, cfg.mainVariant2D};
rows = {};
for run = 1:cfg.nRunsAbl
    [itr, iva, ite] = tn_split(D.meta, cfg, run);
    for g = 1:2
        grp = sprintf('%dD', g);
        for v = 1:numel(V{g})
            vname = V{g}{v};
            f = fullfile(cfg.resDir, sprintf('ABL_%s_%s_run%02d.mat', grp, vname, run));
            if strcmp(vname, mainV{g})
                q = load(fullfile(cfg.resDir, sprintf('R%s_run%02d.mat', grp, run)), 'res');
                r = q.res(strcmp({q.res.model}, ['Turbo_Net-' grp]));
                M = r.M;  tt = r.trainTime;  np = r.nParams;
            elseif exist(f, 'file')
                S = load(f, 'M', 'tt', 'np');  M = S.M;  tt = S.tt;  np = S.np;
            else
                if g == 1, r = tn_trainTurbo1D(D.X, Y, itr, iva, ite, cfg, vname, run);
                else,      r = tn_trainTurbo2D(Sp.U, Y, itr, iva, ite, cfg, vname, run);
                end
                M = r.M;  tt = r.trainTime;  np = r.nParams;
                save(f, 'M', 'tt', 'np');
            end
            rows(end+1, :) = {grp, vname, run, M.acc, M.f1, M.detAcc, np, tt};
            fprintf('  %s %-14s run %d acc = %.4f\n', grp, vname, run, M.acc);
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Group','Variant','Run','Accuracy','MacroF1','DetAcc','nParams','TrainTime_s'});
writetable(T, fullfile(cfg.resDir, 'Ablation.csv'));
end

function step7_tables_figures(cfg)
fprintf('\n=== STEP 7: tables & figures ===\n');
xls = fullfile(cfg.resDir, 'Tables_TurboNet.xlsx');
if exist(xls, 'file'), delete(xls); end
set(groot, 'defaultFigureRenderer', 'painters', 'defaultAxesFontName', 'Times New Roman', ...
    'defaultTextFontName', 'Times New Roman', 'defaultAxesFontSize', 10);
Dm = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'meta', 'fs');
meta = Dm.meta;
lvlTxt = arrayfun(@(k) sprintf('%s (%.1f kg/cm^2)', cfg.levels{k}, cfg.pressure(k)), 1:4, 'UniformOutput', false);
R = [collect(cfg, 'R1D'), collect(cfg, 'R2D')];
models = unique({R.model}, 'stable');
blue = [0.45 0.55 0.70];  red = [0.80 0.25 0.20];

try
    rows = {};
    for m = 1:numel(models)
        r = R(strcmp({R.model}, models{m}));
        rows(end+1, :) = {r(1).group, models{m}, numel(r), ms([r.acc]), ms([r.prec]), ms([r.rec]), ...
            ms([r.f1]), ms([r.kappa], 4), ms([r.detAcc]), ms([r.detSens]), ms([r.detSpec]), ...
            r(1).nParams, mmss([r.trainTime]), mean([r.inferMs])};
    end
    writetable(cell2table(rows, 'VariableNames', {'Group','Model','nRuns','Accuracy','Precision','Recall', ...
        'F1','Kappa','DetAccuracy','DetSensitivity','DetSpecificity','nParams','TrainTime_mmss', ...
        'Infer_ms_per_sample'}), xls, 'Sheet', 'Main');
    mu = arrayfun(@(m) 100*mean([R(strcmp({R.model}, models{m})).acc]), 1:numel(models));
    sd = arrayfun(@(m) 100*std([R(strcmp({R.model}, models{m})).acc]), 1:numel(models));
    fig = figure('Visible', 'off', 'Position', [100 100 700 330]);
    b = bar(mu, 'FaceColor', 'flat');
    for m = 1:numel(models), b.CData(m, :) = blue + (red - blue) * startsWith(models{m}, 'Turbo'); end
    hold on;  errorbar(1:numel(models), mu, sd, 'k.', 'LineWidth', 1);
    for m = 1:numel(models), text(m, mu(m) + sd(m) + 0.5, sprintf('%.2f', mu(m)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
    set(gca, 'XTickLabel', strrep(models, '_', '\_'), 'XTickLabelRotation', 20);
    ylabel('Test accuracy (%)');  ylim([max(0, min(mu - sd) - 10) 100.5]);  grid on;  box on;
    saveFigure(fig, cfg, 'Fig_accuracy_bar');
catch e, warning('TN:main', 'Main table: %s', e.message); end

try
    rows = {};  P = [];  Acc = [];  Inf_ = [];  nm = {};
    for m = 1:numel(models)
        r = R(strcmp({R.model}, models{m}));  np = r(1).nParams;
        rows(end+1, :) = {models{m}, np/1e6, np*4/2^20, mean([r.trainTime]), mean([r.inferMs]), 100*mean([r.acc])};
        if ~isnan(np), P(end+1) = np; Acc(end+1) = 100*mean([r.acc]); Inf_(end+1) = mean([r.inferMs]); nm{end+1} = models{m}; end
    end
    writetable(cell2table(rows, 'VariableNames', {'Model','Params_M','Size_MB','TrainTime_s', ...
        'Infer_ms_per_sample','Accuracy_pct'}), xls, 'Sheet', 'Complexity');
    fig = figure('Visible', 'off', 'Position', [100 100 600 400]);  hold on;
    for k = 1:numel(P)
        scatter(P(k)/1e6, Acc(k), 60 + 40*Inf_(k)/max(Inf_), blue + (red - blue) * startsWith(nm{k}, 'Turbo'), 'filled');
        text(P(k)/1e6 * 1.15, Acc(k), strrep(nm{k}, '_', '\_'), 'FontSize', 9);
    end
    set(gca, 'XScale', 'log');  xlim([min(P)/2e6, max(P)*4e-6]);  grid on;  box on;
    xlabel('Number of parameters (millions, log scale)');  ylabel('Test accuracy (%)');
    saveFigure(fig, cfg, 'Fig_accuracy_vs_params');
catch e, warning('TN:complexity', 'Complexity: %s', e.message); end

try
    pairs = {'Turbo_Net-1D','SVM'; 'Turbo_Net-1D','RF'; 'Turbo_Net-2D','GoogLeNet'; ...
             'Turbo_Net-2D','AlexNet'; 'Turbo_Net-2D','SqueezeNet'; 'Turbo_Net-1D','Turbo_Net-2D'};
    rows = {};
    for k = 1:size(pairs, 1)
        a = [R(strcmp({R.model}, pairs{k, 1})).acc];  c = [R(strcmp({R.model}, pairs{k, 2})).acc];
        if numel(a) < 2 || numel(c) < 2, continue; end
        [p, h] = ranksum(a, c);
        rows(end+1, :) = {pairs{k, 1}, pairs{k, 2}, ms(a), ms(c), p, h};
    end
    if ~isempty(rows)
        writetable(cell2table(rows, 'VariableNames', {'ModelA','ModelB','AccA','AccB','p_value','Significant'}), xls, 'Sheet', 'Wilcoxon');
    end
catch e, warning('TN:wilcoxon', 'Wilcoxon: %s', e.message); end

try
    rows = {};  rows2 = {};
    for m = 1:numel(models)
        r = R(strcmp({R.model}, models{m}));
        PC = cat(3, r.perClass);
        for k = 1:4
            rows(end+1, :) = {models{m}, cfg.levels{k}, cfg.pressure(k), ms(squeeze(PC(k, 1, :))'), ...
                ms(squeeze(PC(k, 2, :))'), ms(squeeze(PC(k, 3, :))')};
        end
        accS = zeros(numel(r), numel(cfg.speeds));
        for i = 1:numel(r)
            sp = meta.speed(r(i).ite);
            for s = 1:numel(cfg.speeds), accS(i, s) = mean(r(i).yPred(sp == cfg.speeds(s)) == r(i).yTrue(sp == cfg.speeds(s))); end
        end
        rows2(end+1, :) = [models(m), arrayfun(@(s) ms(accS(:, s)'), 1:numel(cfg.speeds), 'UniformOutput', false)];
    end
    writetable(cell2table(rows, 'VariableNames', {'Model','Class','Pressure','Precision','Recall','F1'}), xls, 'Sheet', 'PerClass');
    writetable(cell2table(rows2, 'VariableNames', [{'Model'}, arrayfun(@(v) sprintf('Acc_%drpm', v), cfg.speeds, 'UniformOutput', false)]), xls, 'Sheet', 'PerSpeed');
catch e, warning('TN:perclass', 'Per-class/speed: %s', e.message); end

try
    for grp = {'1D', '2D'}
        mm = models(arrayfun(@(m) strcmp(R(find(strcmp({R.model}, models{m}), 1)).group, grp{1}), 1:numel(models)));
        fig = figure('Visible', 'off', 'Position', [100 100 300*numel(mm) 290]);
        tl = tiledlayout(1, numel(mm), 'TileSpacing', 'compact');
        for m = 1:numel(mm)
            r = R(strcmp({R.model}, mm{m}));
            nexttile;
            cc = confusionchart(sum(cat(3, r.cm), 3), cfg.levels, 'Normalization', 'row-normalized');
            cc.Title = strrep(mm{m}, '_', '\_');  cc.FontName = 'Times New Roman';
            cc.XLabel = 'Predicted';  cc.YLabel = 'True';
        end
        title(tl, sprintf('Confusion matrices (%s group, sum of %d runs, row-normalized)', grp{1}, numel(r)));
        saveFigure(fig, cfg, ['Fig_confusion_' grp{1}]);
    end
catch e, warning('TN:confusion', 'Confusion: %s', e.message); end

try
    T = readtable(fullfile(cfg.resDir, 'Noise_robustness.csv'));
    Tn = T;  Tn.SNR = string(Tn.SNR_dB);  Tn.SNR(isinf(Tn.SNR_dB)) = "Clean";
    writetable(Tn(:, {'SNR','Model','Accuracy','MacroF1','DetAcc'}), xls, 'Sheet', 'Noise');
    mm = unique(T.Model, 'stable');  A = zeros(numel(mm), 4);
    for m = 1:numel(mm)
        im = strcmp(T.Model, mm{m});
        A(m, :) = 100 * [T.Accuracy(im & isinf(T.SNR_dB)), T.Accuracy(im & T.SNR_dB == 20), ...
                         T.DetAcc(im & isinf(T.SNR_dB)),   T.DetAcc(im & T.SNR_dB == 20)];
    end
    fig = figure('Visible', 'off', 'Position', [100 100 900 340]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    tt = {'Four-level classification', 'Oil-starvation detection (P0 vs P1-P3)'};
    for p = 1:2
        nexttile;  b = bar(A(:, 2*p-1:2*p));  ylim([0 105]);  grid on;  box on;
        set(gca, 'XTickLabel', strrep(mm, '_', '\_'), 'XTickLabelRotation', 25);
        ylabel('Test accuracy (%)');  title(tt{p});
        for k = 1:2, text(b(k).XEndPoints, b(k).YEndPoints + 2, compose('%.1f', b(k).YData), 'HorizontalAlignment', 'center', 'FontSize', 7); end
        if p == 1, legend({'Clean', 'SNR = 20 dB'}, 'Location', 'southwest'); end
    end
    saveFigure(fig, cfg, 'Fig_noise_robustness');
catch e, warning('TN:noise', 'Noise: %s', e.message); end

try
    T = readtable(fullfile(cfg.resDir, 'Ablation.csv'));
    writetable(groupsummary(T, {'Group', 'Variant'}, {'mean', 'std'}, {'Accuracy', 'MacroF1', 'DetAcc', 'nParams'}), xls, 'Sheet', 'Ablation');
catch e, warning('TN:ablation', 'Ablation: %s', e.message); end

try
    Sp = load(fullfile(cfg.workDir, 'dataset_spec.mat'), 'U', 'fAxis', 'tAxis');
    fig = figure('Visible', 'off', 'Position', [100 100 860 760]);
    tiledlayout(4, 4, 'TileSpacing', 'tight');
    for c = 1:4
        for k = 1:4
            id = find(meta.session == 'S1' & meta.label == cfg.levels{k} & meta.speed == 2000, 1);
            nexttile;  imagesc(Sp.tAxis*1000, Sp.fAxis/1000, Sp.U(:, :, c, id));  axis xy;  colormap(turbo);
            if c == 1, title(lvlTxt{k}); end
            if k == 1, ylabel(sprintf('Ch. %s  f (kHz)', cfg.chanNames{c})); end
            if c == 4, xlabel('Time (ms)'); end
        end
    end
    saveFigure(fig, cfg, 'Fig_data_spectrograms');
catch e, warning('TN:spec', 'Spectrogram figure: %s', e.message); end
try
    Fq = load(fullfile(cfg.workDir, 'dataset_features.mat'));
    fig = figure('Visible', 'off', 'Position', [100 100 900 520]);
    tiledlayout(2, 2, 'TileSpacing', 'compact');
    for c = 1:4
        nexttile;
        boxchart(categorical(meta.speed), Fq.F(:, strcmp(Fq.featNames, [cfg.chanNames{c} '_RMS'])), ...
                 'GroupByColor', meta.label, 'MarkerStyle', 'none');
        xlabel('Engine speed (rpm)');  ylabel('RMS (V)');  title(['Channel ' cfg.chanNames{c}]);  grid on;
        if c == 1, legend(lvlTxt, 'Location', 'northwest', 'FontSize', 7); end
    end
    saveFigure(fig, cfg, 'Fig_data_rms_box');
    sel = {'A_RMS','B_RMS','C_RMS','D_RMS','A_Kurt','B_Kurt','C_Kurt','A_SpCentroid','B_SpCentroid','C_SpCentroid'};
    rows = {};
    for k = 1:4
        id = meta.label == cfg.levels{k};
        rows(end+1, :) = [{cfg.levels{k}, cfg.pressure(k)}, cellfun(@(s) ms(Fq.F(id, strcmp(Fq.featNames, s))', 4), sel, 'UniformOutput', false)];
    end
    writetable(cell2table(rows, 'VariableNames', [{'Level', 'Pressure'}, sel]), xls, 'Sheet', 'Features');
    Dx = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'X');
    L = size(Dx.X, 1);  f = (0:floor(L/2)) * Dm.fs / L;  w = hann(L);
    fig = figure('Visible', 'off', 'Position', [100 100 900 560]);
    tiledlayout(2, 2, 'TileSpacing', 'compact');
    for c = 1:4
        nexttile;  hold on;
        for k = 1:4
            id = find(meta.label == cfg.levels{k} & meta.speed == 2000);
            id = id(1:max(1, floor(numel(id)/200)):end);
            Pw = mean(abs(fft(double(squeeze(Dx.X(:, c, id))) .* w)).^2, 2);
            plot(f/1000, 10*log10(movmean(Pw(1:numel(f)), 31)), 'DisplayName', lvlTxt{k});
        end
        xlim([0 40]);  xlabel('Frequency (kHz)');  ylabel('PSD (dB)');  title(['Channel ' cfg.chanNames{c}]);
        grid on;  box on;  if c == 1, legend('Location', 'northeast', 'FontSize', 7); end
    end
    saveFigure(fig, cfg, 'Fig_data_psd');
catch e, warning('TN:data', 'RMS/PSD figures: %s', e.message); end

try
    N1 = load(fullfile(cfg.netDir, 'run01_1D.mat'), 'netT1D');
    N2 = load(fullfile(cfg.netDir, 'run01_2D.mat'), 'netT2D');
    r1 = load(fullfile(cfg.resDir, 'R1D_run01.mat'), 'ite');  ite = r1.ite;
    if ~exist('Dx', 'var'), Dx = load(fullfile(cfg.workDir, 'dataset_windows.mat'), 'X'); end
    Sp = load(fullfile(cfg.workDir, 'dataset_spec.mat'), 'U');
    f1 = squeeze(minibatchpredict(N1.netT1D, Dx.X(:, :, ite), 'InputDataFormats', "TCB", 'Outputs', 'head_cat'));
    f2 = squeeze(minibatchpredict(N2.netT2D, Sp.U(:, :, :, ite), 'Outputs', 'head_cat'));
    if size(f1, 1) ~= numel(ite), f1 = f1'; end
    if size(f2, 1) ~= numel(ite), f2 = f2'; end
    rng(1);  E = {tsne(double(f1)), tsne(double(f2))};
    fig = figure('Visible', 'off', 'Position', [100 100 860 380]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    col = lines(4);  lab = meta.label(ite);
    tt = {'Turbo\_Net-1D features (t-SNE)', 'Turbo\_Net-2D features (t-SNE)'};
    for p = 1:2
        nexttile;  hold on;
        for k = 1:4, scatter(E{p}(lab == cfg.levels{k}, 1), E{p}(lab == cfg.levels{k}, 2), 6, col(k, :), 'filled', 'DisplayName', lvlTxt{k}); end
        title(tt{p});  box on;  axis tight;
        if p == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    saveFigure(fig, cfg, 'Fig_tsne');
catch e, warning('TN:tsne', 't-SNE: %s', e.message); end
fprintf('Done. Tables: %s\nFigures: %s\n', xls, cfg.figDir);
end

function R = collect(cfg, prefix)
R = struct('group', {}, 'model', {}, 'run', {}, 'acc', {}, 'prec', {}, 'rec', {}, 'f1', {}, ...
    'kappa', {}, 'detAcc', {}, 'detSens', {}, 'detSpec', {}, 'nParams', {}, 'trainTime', {}, ...
    'inferMs', {}, 'cm', {}, 'perClass', {}, 'yPred', {}, 'yTrue', {}, 'ite', {});
fl = dir(fullfile(cfg.resDir, [prefix '_run*.mat']));
for i = 1:numel(fl)
    q = load(fullfile(fl(i).folder, fl(i).name));
    for j = 1:numel(q.res)
        x = q.res(j);  M = x.M;
        R(end+1) = struct('group', prefix(2:3), 'model', x.model, 'run', sscanf(fl(i).name, [prefix '_run%d.mat']), ...
            'acc', M.acc, 'prec', M.precision, 'rec', M.recall, 'f1', M.f1, 'kappa', M.kappa, ...
            'detAcc', M.detAcc, 'detSens', M.detSens, 'detSpec', M.detSpec, 'nParams', x.nParams, ...
            'trainTime', x.trainTime, 'inferMs', x.inferMs, 'cm', M.cm, 'perClass', M.perClass{:, 2:4}, ...
            'yPred', categorical(x.yPred, cfg.levels), 'yTrue', categorical(q.yTrue, cfg.levels), 'ite', q.ite);
    end
end
end

function s = ms(x, d)
if nargin < 2, s = sprintf('%.2f ± %.2f', 100*mean(x), 100*std(x));
else,          s = sprintf(['%.' num2str(d) 'g ± %.' num2str(d) 'g'], mean(x), std(x));
end
end

function s = mmss(t)
f = @(x) sprintf('%02d:%02d', floor(x/60), round(mod(x, 60)));
s = sprintf('%s ± %s', f(mean(t)), f(std(t)));
end

function saveFigure(fig, cfg, name)
fig.Visible = 'on';  drawnow;
f = fullfile(cfg.figDir, [name '.png']);
try, exportgraphics(fig, f, 'Resolution', 300); catch, print(fig, f, '-dpng', '-r300'); end
close(fig);
end

function [itr, iva, ite] = tn_split(meta, cfg, run)
rng(cfg.seed0 + run, 'twister');
itr = [];  iva = [];  ite = [];
cats = categories(meta.label);
for k = 1:numel(cats)
    id = find(meta.label == cats{k});  id = id(randperm(numel(id)));
    n = numel(id);  ntr = round(cfg.split(1) * n);  nva = round(cfg.split(2) * n);
    itr = [itr; id(1:ntr)];  iva = [iva; id(ntr+1:ntr+nva)];  ite = [ite; id(ntr+nva+1:end)];
end
itr = itr(randperm(numel(itr)));
end

function opts = tn_trainOpts(cfg, valData, lr, maxEpochs, nTrain, fmt, patience)
if nargin < 6, fmt = []; end
if nargin < 7 || isempty(patience), patience = cfg.train.patience; end
args = {'InitialLearnRate', lr, 'MaxEpochs', maxEpochs, 'MiniBatchSize', cfg.train.miniBatch, ...
        'Shuffle', 'every-epoch', 'ValidationData', valData, ...
        'ValidationFrequency', max(1, floor(nTrain / cfg.train.miniBatch)), 'ValidationPatience', patience, ...
        'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', cfg.train.dropPeriod, ...
        'LearnRateDropFactor', cfg.train.dropFactor, 'L2Regularization', cfg.train.l2, ...
        'Metrics', 'accuracy', 'Verbose', cfg.train.verbose, 'VerboseFrequency', 50, ...
        'Plots', 'none', 'ExecutionEnvironment', cfg.train.exec};
if ~isempty(fmt), args = [args, {'InputDataFormats', fmt}]; end
try
    opts = trainingOptions('adam', args{:}, 'OutputNetwork', 'best-validation');
catch
    opts = trainingOptions('adam', args{:}, 'OutputNetwork', 'best-validation-loss');
end
end

function [r, net] = tn_trainTurbo1D(X, Y, itr, iva, ite, cfg, variant, run)
classes = categorical(cfg.levels, cfg.levels);
Xtr = X(:, :, itr);
net = tn_buildTurboNet1D(cfg.winLen, numel(cfg.levels), variant, mean(Xtr, [1 3]), std(Xtr, 0, [1 3]));
opts = tn_trainOpts(cfg, {X(:, :, iva), Y(iva)}, cfg.train.lrScratch, cfg.train.maxEpochs, ...
                    numel(itr), "TCB", cfg.train.patienceScratch);
rng(cfg.seed0 + 100 + run);
t = tic;  [net, info] = trainnet(Xtr, Y(itr), net, 'crossentropy', opts);  tt = toc(t);
clear Xtr
t = tic;  sc = minibatchpredict(net, X(:, :, ite), 'InputDataFormats', "TCB", 'MiniBatchSize', 256);  ti = toc(t);
yp = scores2label(sc, cfg.levels);
r = tn_pack('Turbo_Net-1D', tn_metrics(Y(ite), yp, classes), tt, 1000*ti/numel(ite), tn_nparams(net), yp, ...
            struct('train', info.TrainingHistory, 'val', info.ValidationHistory), struct('variant', variant));
end

function [r, net] = tn_trainTurbo2D(U, Y, itr, iva, ite, cfg, variant, run, specMs)
if nargin < 9, specMs = 0; end
classes = categorical(cfg.levels, cfg.levels);
net = tn_buildTurboNet2D([size(U, 1) size(U, 2) 4], numel(cfg.levels), variant);
opts = tn_trainOpts(cfg, {U(:, :, :, iva), Y(iva)}, cfg.train.lrScratch, cfg.train.maxEpochs, ...
                    numel(itr), [], cfg.train.patienceScratch);
rng(cfg.seed0 + 200 + run);
t = tic;  [net, info] = trainnet(U(:, :, :, itr), Y(itr), net, 'crossentropy', opts);  tt = toc(t);
t = tic;  sc = minibatchpredict(net, U(:, :, :, ite), 'MiniBatchSize', 256);  ti = toc(t);
yp = scores2label(sc, cfg.levels);
r = tn_pack('Turbo_Net-2D', tn_metrics(Y(ite), yp, classes), tt, 1000*ti/numel(ite) + specMs, tn_nparams(net), yp, ...
            struct('train', info.TrainingHistory, 'val', info.ValidationHistory), struct('variant', variant));
end

function M = tn_metrics(yt, yp, classes)
yt = categorical(yt, categories(classes));  yp = categorical(yp, categories(classes));
cm = confusionmat(yt, yp, 'Order', categories(classes));
tp = diag(cm);  prec = tp ./ max(sum(cm, 1)', 1);  rec = tp ./ max(sum(cm, 2), 1);
f1 = 2 * prec .* rec ./ max(prec + rec, eps);  n = sum(cm(:));
M.acc = sum(tp) / n;  M.precision = mean(prec);  M.recall = mean(rec);  M.f1 = mean(f1);
pe = sum(sum(cm, 1) .* sum(cm, 2)') / n^2;  M.kappa = (M.acc - pe) / (1 - pe);
M.perClass = table(categories(classes), prec, rec, f1, 'VariableNames', {'Class','Precision','Recall','F1'});
M.cm = cm;
c0 = categories(classes);  tN = yt == c0{1};  pN = yp == c0{1};
TP = sum(~tN & ~pN);  TN = sum(tN & pN);  FP = sum(tN & ~pN);  FN = sum(~tN & pN);
M.detAcc = (TP + TN) / n;  M.detSens = TP / max(TP + FN, 1);  M.detSpec = TN / max(TN + FP, 1);
end

function r = tn_pack(model, M, trainTime, inferMs, nParams, yPred, history, extra)
if nargin < 7, history = []; end
if nargin < 8, extra = struct(); end
r = struct('model', model, 'M', M, 'trainTime', trainTime, 'inferMs', inferMs, ...
           'nParams', nParams, 'yPred', yPred, 'history', history, 'extra', extra);
end

function n = tn_nparams(net)
n = sum(cellfun(@numel, net.Learnables.Value));
end

function [useRG, useMS, useSE, useD] = tn_variantFlags(variant)
switch variant
    case 'lean',          f = [0 1 0 1];
    case 'lean_noMS',     f = [0 0 0 1];
    case 'lean_noD',      f = [0 1 0 0];
    case 'lean_noMS_noD', f = [0 0 0 0];
    case 'full',          f = [1 1 1 1];
    otherwise, error('Unknown variant: %s', variant);
end
useRG = f(1) == 1;  useMS = f(2) == 1;  useSE = f(3) == 1;  useD = f(4) == 1;
end

function net = tn_buildTurboNet1D(L, nClass, variant, mu, sd)
if nargin < 4, mu = zeros(4, 1); sd = ones(4, 1); end
[useRG, useMS, useSE, useD] = tn_variantFlags(variant);
stemF = 16;  chans = [64 96 128 128];  k = 7;
gate = @(F, g) dlarray(stripdims(F) .* stripdims(g), dims(F));
net = dlnetwork;
net = addLayers(net, sequenceInputLayer(4, 'MinLength', L, 'Name', 'input', ...
    'Normalization', 'zscore', 'Mean', double(mu(:)), 'StandardDeviation', double(sd(:))));
if ~useRG && useD
    mainIn = 'input';
else
    net = addLayers(net, functionLayer(@(X) X(1:3,:,:), 'Formattable', true, 'Acceleratable', true, 'Name', 'turbo_ABC'));
    net = connectLayers(net, 'input', 'turbo_ABC');  mainIn = 'turbo_ABC';
end
if useMS
    ks = [15 63 255];
    net = addLayers(net, depthConcatenationLayer(3, 'Name', 'stem_cat'));
    for i = 1:3
        nm = sprintf('stem_k%d', ks(i));
        net = addLayers(net, convolution1dLayer(ks(i), stemF, 'Stride', 4, 'Padding', 'same', 'Name', nm));
        net = connectLayers(net, mainIn, nm);  net = connectLayers(net, nm, sprintf('stem_cat/in%d', i));
    end
    last = 'stem_cat';
else
    net = addLayers(net, convolution1dLayer(63, 3*stemF, 'Stride', 4, 'Padding', 'same', 'Name', 'stem_k63'));
    net = connectLayers(net, mainIn, 'stem_k63');  last = 'stem_k63';
end
net = addLayers(net, [batchNormalizationLayer('Name', 'stem_bn'); reluLayer('Name', 'stem_relu')
                      maxPooling1dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'stem_pool')]);
net = connectLayers(net, last, 'stem_bn');  last = 'stem_pool';  cin = 3*stemF;
for s = 1:numel(chans)
    [net, last] = resBlock(net, last, cin, chans(s), 1, k, useSE, sprintf('s%d', s), gate);
    cin = chans(s);
end
if useRG
    net = addLayers(net, [functionLayer(@(X) X(4,:,:), 'Formattable', true, 'Acceleratable', true, 'Name', 'ref_D')
        convolution1dLayer(63, 16, 'Stride', 4, 'Padding', 'same', 'Name', 'ref_c1'); batchNormalizationLayer('Name', 'ref_bn1'); reluLayer('Name', 'ref_r1')
        maxPooling1dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_pool')
        convolution1dLayer(9, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_c2'); batchNormalizationLayer('Name', 'ref_bn2'); reluLayer('Name', 'ref_r2')
        convolution1dLayer(9, 64, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_c3'); batchNormalizationLayer('Name', 'ref_bn3'); reluLayer('Name', 'ref_r3')
        globalAveragePooling1dLayer('Name', 'ref_gap')]);
    net = connectLayers(net, 'input', 'ref_D');
    net = addLayers(net, globalAveragePooling1dLayer('Name', 'rg_gapF'));
    [net, last] = gatedFusion(net, last, cin, gate);
end
net = addHead(net, last, useRG, nClass, @globalAveragePooling1dLayer, @globalMaxPooling1dLayer);
end

function net = tn_buildTurboNet2D(inSize, nClass, variant)
[useRG, useMS, useSE, useD] = tn_variantFlags(variant);
stemF = 16;  chans = [64 96 128 128];
gate = @(F, g) dlarray(stripdims(F) .* reshape(stripdims(g), 1, 1, size(g, 1), []), dims(F));
net = dlnetwork;
net = addLayers(net, imageInputLayer(inSize, 'Name', 'input', 'Normalization', 'rescale-zero-one', 'Min', 0, 'Max', 255));
if ~useRG && useD
    mainIn = 'input';
else
    net = addLayers(net, functionLayer(@(X) X(:,:,1:3,:), 'Formattable', true, 'Acceleratable', true, 'Name', 'turbo_ABC'));
    net = connectLayers(net, 'input', 'turbo_ABC');  mainIn = 'turbo_ABC';
end
if useMS
    ks = {[3 3], [9 3], [3 9]};
    net = addLayers(net, depthConcatenationLayer(3, 'Name', 'stem_cat'));
    for i = 1:3
        nm = sprintf('stem_%dx%d', ks{i}(1), ks{i}(2));
        net = addLayers(net, convolution2dLayer(ks{i}, stemF, 'Padding', 'same', 'Name', nm));
        net = connectLayers(net, mainIn, nm);  net = connectLayers(net, nm, sprintf('stem_cat/in%d', i));
    end
    last = 'stem_cat';
else
    net = addLayers(net, convolution2dLayer(5, 3*stemF, 'Padding', 'same', 'Name', 'stem_5x5'));
    net = connectLayers(net, mainIn, 'stem_5x5');  last = 'stem_5x5';
end
net = addLayers(net, [batchNormalizationLayer('Name', 'stem_bn'); reluLayer('Name', 'stem_relu')
                      maxPooling2dLayer(2, 'Stride', 2, 'Name', 'stem_pool')]);
net = connectLayers(net, last, 'stem_bn');  last = 'stem_pool';  cin = 3*stemF;
for s = 1:numel(chans)
    [net, last] = resBlock(net, last, cin, chans(s), 2, 3, useSE, sprintf('s%d', s), gate);
    cin = chans(s);
end
if useRG
    net = addLayers(net, [functionLayer(@(X) X(:,:,4,:), 'Formattable', true, 'Acceleratable', true, 'Name', 'ref_D')
        convolution2dLayer(5, 16, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_c1'); batchNormalizationLayer('Name', 'ref_bn1'); reluLayer('Name', 'ref_r1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'ref_pool')
        convolution2dLayer(3, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_c2'); batchNormalizationLayer('Name', 'ref_bn2'); reluLayer('Name', 'ref_r2')
        convolution2dLayer(3, 64, 'Stride', 2, 'Padding', 'same', 'Name', 'ref_c3'); batchNormalizationLayer('Name', 'ref_bn3'); reluLayer('Name', 'ref_r3')
        globalAveragePooling2dLayer('Name', 'ref_gap')]);
    net = connectLayers(net, 'input', 'ref_D');
    net = addLayers(net, globalAveragePooling2dLayer('Name', 'rg_gapF'));
    [net, last] = gatedFusion(net, last, cin, gate);
end
net = addHead(net, last, useRG, nClass, @globalAveragePooling2dLayer, @globalMaxPooling2dLayer);
end

function [net, out] = resBlock(net, in, ~, cout, nd, k, useSE, p, gate)
conv = @(kk, c, varargin) convLayer(nd, kk, c, varargin{:});
net = addLayers(net, [conv(k, cout, 'Stride', 2, 'Padding', 'same', 'Name', [p '_c1'])
                      batchNormalizationLayer('Name', [p '_bn1']); reluLayer('Name', [p '_r1'])
                      conv(k, cout, 'Padding', 'same', 'Name', [p '_c2'])
                      batchNormalizationLayer('Name', [p '_bn2'])]);
net = connectLayers(net, in, [p '_c1']);  last = [p '_bn2'];
if useSE
    gap = {@globalAveragePooling1dLayer, @globalAveragePooling2dLayer};
    net = addLayers(net, [gap{nd}('Name', [p '_se_gap']); fullyConnectedLayer(max(4, round(cout/8)), 'Name', [p '_se_fc1'])
                          reluLayer('Name', [p '_se_relu']); fullyConnectedLayer(cout, 'Name', [p '_se_fc2'])
                          sigmoidLayer('Name', [p '_se_sig'])]);
    net = addLayers(net, functionLayer(gate, 'NumInputs', 2, 'Formattable', true, 'Acceleratable', true, 'Name', [p '_se_gate']));
    net = connectLayers(net, last, [p '_se_gap']);  net = connectLayers(net, last, [p '_se_gate/in1']);
    net = connectLayers(net, [p '_se_sig'], [p '_se_gate/in2']);  last = [p '_se_gate'];
end
net = addLayers(net, additionLayer(2, 'Name', [p '_add']));
net = addLayers(net, reluLayer('Name', [p '_out']));
net = connectLayers(net, last, [p '_add/in1']);
net = addLayers(net, [conv(1, cout, 'Stride', 2, 'Padding', 'same', 'Name', [p '_sc'])
                      batchNormalizationLayer('Name', [p '_sc_bn'])]);
net = connectLayers(net, in, [p '_sc']);  net = connectLayers(net, [p '_sc_bn'], [p '_add/in2']);
net = connectLayers(net, [p '_add'], [p '_out']);  out = [p '_out'];
end

function L = convLayer(nd, k, c, varargin)
if nd == 1, L = convolution1dLayer(k, c, varargin{:}); else, L = convolution2dLayer(k, c, varargin{:}); end
end

function [net, last] = gatedFusion(net, last, cin, gate)
net = addLayers(net, [depthConcatenationLayer(2, 'Name', 'rg_cat'); fullyConnectedLayer(32, 'Name', 'rg_fc1')
                      reluLayer('Name', 'rg_relu'); fullyConnectedLayer(cin, 'Name', 'rg_fc2'); sigmoidLayer('Name', 'rg_sig')]);
net = addLayers(net, functionLayer(gate, 'NumInputs', 2, 'Formattable', true, 'Acceleratable', true, 'Name', 'rg_gate'));
net = connectLayers(net, last, 'rg_gapF');  net = connectLayers(net, 'rg_gapF', 'rg_cat/in1');
net = connectLayers(net, 'ref_gap', 'rg_cat/in2');  net = connectLayers(net, last, 'rg_gate/in1');
net = connectLayers(net, 'rg_sig', 'rg_gate/in2');  last = 'rg_gate';
end

function net = addHead(net, last, useRG, nClass, gapFcn, gmpFcn)
net = addLayers(net, gapFcn('Name', 'head_gap'));
net = addLayers(net, gmpFcn('Name', 'head_gmp'));
net = addLayers(net, [depthConcatenationLayer(2 + useRG, 'Name', 'head_cat'); dropoutLayer(0.3, 'Name', 'head_drop')
                      fullyConnectedLayer(nClass, 'Name', 'fc_out'); softmaxLayer('Name', 'softmax')]);
net = connectLayers(net, last, 'head_gap');  net = connectLayers(net, last, 'head_gmp');
net = connectLayers(net, 'head_gap', 'head_cat/in1');  net = connectLayers(net, 'head_gmp', 'head_cat/in2');
if useRG, net = connectLayers(net, 'ref_gap', 'head_cat/in3'); end
net = initialize(net);
end
