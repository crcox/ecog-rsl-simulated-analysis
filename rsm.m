function v = rsm(x, distance)
    arguments
        x (:,:) double {mustBeNumeric}
        distance (1,1) string {mustBeTextScalar} = "cosine"
    end
    v = 1 - pdist(x, distance);
end