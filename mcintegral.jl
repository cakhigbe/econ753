using Statistics

function mc_integral(f, a, b, n)
# Generate random points in interval [a,b]
x = a .+ (b-a) * rand(n)

# Compute function values and scale by interval width
y = f.(x) * (b-a)

# Return mean and standard error
return mean(y), std(y)/sqrt(n)
end

# Test with f(x) = x^2 on [0,1]
f(x) = x^2
n = 100000
result, error = mc_integral(f, 0, 1, n)

println("Integral of x^2 from 0 to 1:")
println("Monte Carlo: $result ± $error")
println("Exact: $(1/3)")
