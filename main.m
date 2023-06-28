load("metadata_raw_2021May05.mat")
S = metadata(1).targets(3).target;
C = u(:, 1:3) * sqrt(s(1:3, 1:3));
z = tril(true(size(S)), -1);
target_rsm = S(z);

nexamples = 100;
nfeatures = 24;
X0 = struct( ...
    'categorical', [repelem([1;-1], nexamples/2, nfeatures/2), rand_matchrange([nexamples, nfeatures/2], [-1;1])], ...
    'dim1', [repmat(C(:, 1), 1, nfeatures/2), rand_matchrange([nexamples, nfeatures/2], C(:, 1))], ...
    'dim2', [repmat(C(:, 2), 1, nfeatures/2), rand_matchrange([nexamples, nfeatures/2], C(:, 2))], ...
    'dim3', [repmat(C(:, 3), 1, nfeatures/2), rand_matchrange([nexamples, nfeatures/2], C(:, 3))], ...
    'full', [repmat(C, 1, (nfeatures/2)/3), rand_matchrange([nexamples, 3], C), rand_matchrange([nexamples, 3], C), rand_matchrange([nexamples, 3], C), rand_matchrange([nexamples, 3], C)] ...
);

RSA = zeros(11, 5, 20);
for j = 1:20
    for i = 0:10
        if i > 0
            X = structfun(@(x) x + rand_matchrange([nexamples, nfeatures], [-i;i]./2), X0, 'UniformOutput', false);
            NSM = structfun(@(x) rsm(x, 'cosine'), X, 'UniformOutput', false);
        else
            NSM = structfun(@(x) rsm(x, 'cosine'), X0, 'UniformOutput', false); 
        end
        RSA(i+1, :, j) = structfun(@(x) corr(x', target_rsm), NSM);
    end
end

plot(0:10, mean(RSA, 3))
ylim([-.05, 1]);
yline(0)
legend(["Binary", "1st only", "2nd only", "3rd only", "All"])


cv = cvpartition(repelem([1,2], 50), "KFold", 10);
RSL = zeros(11, 5, 3, 20);
for k = 1:20
    for j = 1:3
        for i = 0:10
            if i > 0
                X = structfun(@(x) x + rand_matchrange([nexamples, nfeatures], [-i;i]./2), X0, 'UniformOutput', false);
            else
                X = X0;
            end
        end

        for i = 1:10
            Cz = structfun(@(x) zeros(size(C)), X, 'UniformOutput', false);
            z = training(cv, i) == 1;
            m = structfun(@(x) glmfit(x(z,:), C(z, j)), X, 'UniformOutput', false);

        RSA(:, i+1, j, k) = structfun(@(x) corr(x', target_rsm), NSM);
    end
end



function v = rsm(x, distance)
    arguments
        x (:,:) double {mustBeNumeric}
        distance (1,1) string {mustBeTextScalar} = "cosine"
    end
    v = 1 - pdist(x, distance);
end