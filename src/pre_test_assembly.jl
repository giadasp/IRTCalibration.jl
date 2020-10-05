function pre_test_CPLEX(
    n_items,
    T_pt,
    guessed_difficulty,
    alphas,
    n_pt,
    n_couples,
    f1,
    f2,
    ol_max,
)
    m = JuMP.Model(
        solver = CPLEX.CplexSolver(
            CPX_PARAM_TILIM = 100,
            CPX_PARAM_SOLNPOOLAGAP = 0,
            CPX_PARAM_SOLNPOOLINTENSITY = 4,
            CPX_PARAM_POPULATELIM = 10,
        ),
    )
    JuMP.@variable(m, x[i = 1:n_items, t = 1:T_pt] , Bin)
    JuMP.@variable(m, y[i = 1:n_items, g = 1:n_couples], Bin)
    #length
    JuMP.@constraint(m, [t = 1:T_pt], sum(x[i, t] for i = 1:n_items) == n_pt)
    #respect the Distributions.Distribution of dfficulty 25% of low difficulty items, 50% of medium items, 25% of high items.
    for g = 1:size(alphas, 1)
        JuMP.@constraint(
            m,
            [t = 1:T_pt],
            sum(x[i, t] for i in findall(guessed_difficulty .== g)) <= alphas[g] * n_pt + 1
        )
    end
    #item use min=1 use all items
    JuMP.@constraint(m, [i = 1:n_items], sum(x[i, t] for t = 1:T_pt) >= 1)
    #overlap, 5 btw each pair
    JuMP.@constraint(m, [g = 1:n_couples], sum(y[i, g] for i = 1:n_items) == ol_max[g])
    JuMP.@constraint(
        m,
        [g = 1:n_couples, i = 1:n_items],
        2 * y[i, g] <= x[i, f1[g]] + x[i, f2[g]]
    )
    JuMP.@constraint(
        m,
        [g = 1:n_couples, i = 1:n_items],
        y[i, g] >= x[i, f1[g]] + x[i, f2[g]] - 1
    )
    JuMP.solve(m)
    Design = (JuMP.getvalue(x))
    return Design
end
function overlap_pairs(T_pt, overlap_matrix)
    n_couples = binomial(T_pt, 2)
    couples = Vector{Vector{Int64}}(undef, n_couples)
    p = 1
    for t1 = 1:T_pt
        for t2 = t1+1:T_pt
            couples[p] = [t1, t2]
            p += 1
        end
    end
    ol_max = Array{Int64,1}(undef, n_couples)
    f1 = [couples[g][1] for g = 1:n_couples]
    f2 = [couples[g][2] for g = 1:n_couples]
    for g = 1:n_couples
        ol_max[g] = overlap_matrix[f1[g], f2[g]]
    end
    return (n_couples, f1, f2, ol_max)
end
