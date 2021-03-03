function my_rescale(
    X::Vector{Float64},
    Wk::Vector{Float64},
    observed::Vector{Float64},
)
    #println("theta had mean ",Wk'*X)
    #println("theta had std ", sqrt(Wk'*(X.^2)))
    #if abs(observed[1])>1e-3
    X = X .- observed[1]
    #end
    #if observed[2]<0.99
    X = X ./ observed[2]
    #end
    #println("theta has mean ",Wk'*X)
    #println("theta has std ", sqrt(Wk'*((X).^2)))
    X = vcat(X[1] - (X[2] - X[1]), X)
    X = vcat(X, X[end] + (X[2] - X[1]))
    Wk = vcat(zero(Float64), Wk)
    Wk = vcat(Wk, zero(Float64))
    # Wk=round.(Wk,digits=5)
    Wk = Wk ./ sum(Wk)
    return X, Wk
end

function cubic_spline_int(X::Vector{Float64}, NewX::Vector{Float64}, Wk::Vector{Float64})
    bw = abs(NewX[2] - NewX[1])
    K = size(X, 1)
    ok = 0
    #Wk2=copy(Wk)
    # while ok==0
    # 	if NewX[1]>X[1]
    # 		NewX=vcat(NewX[1]-bw,NewX)
    # 		Wk2=vcat(zero(Float64),Wk2)
    # 	end
    # 	if NewX[end]<X[end]
    # 		NewX=vcat(NewX,NewX[end]+bw)
    # 		Wk2=vcat(Wk2,zero(Float64))
    # 	end
    # 	if NewX[1]<=X[1] && NewX[end]>=X[end]
    # 		ok=1
    # 	end
    # end
    # Wk=Wk./sum(Wk)
    #interp_cubic = CubicSplineInterpolation(NewX, Wk2)
    #BSpline(Quadratic(Line(OnGrid()))) or BSpline(Cubic(Line(OnGrid()))) or BSpline(Quadratic(Line(OnGrid())))
    #type Constant, Linear, Quadratic, and Cubic
    #for boundaries can be OnGrid (at edge of the points), OnCell (hal way between the points),
    #boundary conditions Flat, Line, Natural, Free, Periodic and Reflect
    #not bad BSpline(Cubic(Flat(OnGrid())))), NewX), Flat()
    #better hist but higher RMSE BSpline(Cubic(Line(OnGrid())))), NewX), Flat()
    #separate the knots in areas
    #main area
    # der=[abs(Wk2[k]-Wk2[k+1])/bw for k=1:(K-1)]
    #println(Wk)
    # foundInf=0
    # foundSup=0
    mainArealiminf = 1
    mainArealimsup = K + 2
    # k=1
    # while (foundInf+foundSup<2 && k<K)
    # 	#look for inf
    # 	if foundInf<1
    # 		if (((Wk[k+1]-Wk[k]))/bw)>1e-2
    # 			mainArealiminf=max(k+1,mainArealiminf)
    # 			foundInf=1
    # 		end
    # 	end
    # 	if foundSup<1
    # 		if ((Wk[K-k+1]-Wk[K-k]))/bw<-1e-2
    # 			mainArealimsup=min(K-k+2,mainArealimsup)
    # 			foundSup=1
    # 		end
    # 	end
    # 	#llof for sup
    # 	k+=1
    # end
    # #println(mainArealiminf," ",mainArealimsup)
    # Wk=Wk[mainArealiminf:mainArealimsup]
    NewX = NewX[mainArealiminf]:bw:(NewX[mainArealiminf]+(bw*(size(Wk, 1)-1)))
    # K2=size(NewX,1)
    # idx=collect(1:Int(trunc((K2)/(min(8,K2-1)))):K2)[2:(end-1)] #take only 5 points as knots
    # NewX=range(NewX[idx[1]],stop=NewX[idx[end]],length=size(idx,1))
    interp_cubic = Interpolations.extrapolate(
        Interpolations.scale(
            Interpolations.interpolate(
                Wk,
                Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))),
            ),
            NewX,
        ),
        Interpolations.Line(),
    )
    #interp_linear=Interpolations.extrapolate(Interpolations.scale(Interpolations.interpolate(Wk, Interpolations.BSpline(Interpolations.Linear())), NewX), Interpolations.Line())

    Wk_cubic = zeros(K)
    #Wk_linear=similar(Wk_cubic)
    for k = 1:K
        #Wk_linear[k]=interp_linear(X[k])
        Wk_cubic[k] = interp_cubic(X[k])
    end
    #println(Wk_linear)
    #println(Wk_cubic)
    #Wk=sum([Wk_linear,Wk_cubic],dims=1)[1]./2
    #Wk=max.(Wk_linear,Wk_cubic)
    #println(Wk)
    #Wk=clamp.(Wk,zero(Float64),one(Float64))
    #Wk=Wk.+maximum(abs.(Wk[findall(Wk.<0)]))
    Wk = clamp.(Wk_cubic, 1e-20, one(Float64))
    #println(Wk)
    Wk = Wk ./ sum(Wk)

    println(Wk)
    return Wk::Vector{Float64}
end
#Dierckx package
# function cubic_spline_direct(X::Vector{Float64}, NewX::Vector{Float64}, Wk::Vector{Float64})
#     bw = abs(NewX[3] - NewX[2])
#     K = size(X, 1)
#     ok = 0
#     #Wk2=copy(Wk)
#     # while ok==0
#     # 	if NewX[1]>X[1]
#     # 		NewX=vcat(NewX[1]-bw,NewX)
#     # 		Wk2=vcat(zero(Float64),Wk2)
#     # 	end
#     # 	if NewX[end]<X[end]
#     # 		NewX=vcat(NewX,NewX[end]+bw)
#     # 		Wk2=vcat(Wk2,zero(Float64))
#     # 	end
#     # 	if NewX[1]<=X[1] && NewX[end]>=X[end]
#     # 		ok=1
#     # 	end
#     # end
#     # Wk=Wk./sum(Wk)
#     #interp_cubic = CubicSplineInterpolation(NewX, Wk2)
#     #BSpline(Quadratic(Line(OnGrid()))) or BSpline(Cubic(Line(OnGrid()))) or BSpline(Quadratic(Line(OnGrid())))
#     #type Constant, Linear, Quadratic, and Cubic
#     #for boundaries can be OnGrid (at edge of the points), OnCell (hal way between the points),
#     #boundary conditions Flat, Line, Natural, Free, Periodic and Reflect
#     #not bad BSpline(Cubic(Flat(OnGrid())))), NewX), Flat()
#     #better hist but higher RMSE BSpline(Cubic(Line(OnGrid())))), NewX), Flat()
#     #separate the knots in areas
#     #main area
#     # der=[abs(Wk2[k]-Wk2[k+1])/bw for k=1:(K-1)]
#     println(Wk)
#     # foundInf=0
#     # foundSup=0
#     # mainArealiminf=1
#     # mainArealimsup=K+2
#     # k=1
#     # while (foundInf+foundSup<2 && k<K)
#     # 	#look for inf
#     # 	if foundInf<1
#     # 		if ((abs(Wk[k]-Wk[k+1]))/bw)>1e-3
#     # 			mainArealiminf=max(k-2,mainArealiminf)
#     # 			foundInf=1
#     # 		end
#     # 	end
#     # 	if foundSup<1
#     # 		if (abs(Wk[K-k+1]-Wk[K-k]))/bw>1e-3
#     # 			mainArealimsup=min(K-k+3,mainArealimsup)
#     # 			foundSup=1
#     # 		end
#     # 	end
#     # 	#llof for sup
#     # 	k+=1
#     # end
#     # # mainArealiminf=max(findfirst(der.>1e-4)-1,1)
#     # # mainArealimsup=min(findlast(der.>1e-4)+2,K)
#     # Wk=Wk[mainArealiminf:mainArealimsup]
#     # NewX=NewX[mainArealiminf:mainArealimsup]
#     K2 = size(NewX, 1)

#     #interp_cubic=Dierckx.Spline1D(NewXMainArea, WkMainArea; w=(Distributions.pdf.(Distributions.Normal(0,1),NewXMainArea)./(sum(Distributions.pdf.(Distributions.Normal(0,1),NewXMainArea)))), k=3, bc="zero", s=0.00)
#     idx = collect(1:Int(trunc((K2) / (min(6, K2 - 1)))):K2)[2:(end-1)] #take only 5 points as knots
#     interp_cubic =
#         Dierckx.Spline1D(NewX, Wk, NewX[3:K2-2]; w = ones(K2), k = 1, bc = "nearest")#, s=0.00) #good with 11 points

#     Wk = zeros(K)
#     for k = 1:K
#         Wk[k] = Dierckx.evaluate(interp_cubic, X[k])
#     end
#     #println(Wk)
#     Wk = clamp.(Wk, 1e-10, one(Float64))
#     #Wk=Wk.+maximum(abs.(Wk[findall(Wk.<0)]))

#     Wk = Wk ./ sum(Wk)
#     println(Wk)

#     return Wk::Vector{Float64}
# end
