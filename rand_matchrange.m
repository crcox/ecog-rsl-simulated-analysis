function y = rand_matchrange(sz, x)
    if size(x, 2) > 1
        assert(sz(2) == size(x, 2))
    end
    y = rand(sz);
    y = bsxfun(@minus, y, min(y));
    y = bsxfun(@rdivide, y, max(y));
    y = bsxfun(@times, y, peak2peak(x));
    y = bsxfun(@plus, y, min(x));
end