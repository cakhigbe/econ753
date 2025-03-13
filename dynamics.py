import numpy as np
from scipy.optimize import minimize

np.random.seed(42)

# parameters
mu = -1 # loss per age
R = -3 # replacement cost
beta = 0.9 # discount factor
gamma = 0.5775 # euler const

T = 20000 # time periods

# state variables
age = np.array([1,2,3,4,5])

############################################################
# functions                                                #
############################################################
# contraction mapping: value function iteration
tol = 1e-6
max_iter = 1000
def contraction_mapping(mu, R):
    V = np.zeros(len(age)) # initialize value function
    for j in range(max_iter):
        V_new = np.zeros(len(age))

        for i, a in enumerate(age):
            # choice specific conditional value functions
            V_0 = mu * a + beta * V[min(i+1,4)] # keep
            V_1 = R + beta * V[0] # replace

            # update bellman using logsum
            V_new[i] = gamma + np.log(np.exp(V_0) + np.exp(V_1))

        # check convergence
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new.copy()
    return V
V = contraction_mapping(mu, R) # solve

# loglikelihood function for NFP
def llh(params):
    mu, R = params
    V = contraction_mapping(mu, R)
    log_like = 0

    for t in range(T):
        a_t = states[t]
        i_t = actions[t]

        V_0 = mu * a_t + beta * V[min(a_t+1, 4)]  # keep
        V_1 = R + beta * V[0]  # replace

        prob_rep = np.exp(V_1)/(np.exp(V_0)+np.exp(V_1))

        if i_t == 1:
            log_like += np.log(prob_rep)
        else:
            log_like += np.log(1-prob_rep)

    return -log_like # negative for minimize

# procedure for forward simulation of V0 and V1 for CCP
def fwd_sim(P_hat, F_U, mu, R):
    V0 = np.zeros(5)
    V1 = np.zeros(5)
    for j in range(max_iter):
        V0_new = mu * age + beta * F_U @ V0
        V1_new = -R + beta * V0[0] * np.ones(5)

        if np.max(np.abs(V0_new - V0)) < tol and np.max(np.abs(V1_new - V1)) < tol:
            break
        V0, V1 = V0_new, V1_new
    return V0, V1
############################################################
############################################################


# indifference condition at a_t=2
V_02 = mu * 2 + beta * V[2] # keep
V_12 = R + beta * V[0] # replace

eps_diff = V_12 - V_02

# prob of replacement at a_t=2
# using logit choice probs
prob_replace = np.exp(V_12)/(np.exp(V_02)+np.exp(V_12))

# value at state a_t=4, eps_0=1, eps_1=1.5
V_04 = mu * 4 + beta * V[4] + 1 # keep
V_14 = V_12 + 1.5 # replace

V_4 = max(V_04, V_14)

# simulate data
states = np.zeros(T, dtype=int) # machine ages
actions = np.zeros(T, dtype=int) # observable choices

state = 1 # initial state

for t in range(T):
    # compute choice probs
    V_0 = mu * state + beta * V[min(state, 4)]  # keep, zero-index so + 1 implicit
    V_1 = R + beta * V[0]  # replace
    prob_rep = np.exp(V_1)/(np.exp(V_0)+np.exp(V_1))

    # choice based on probabilities
    action = np.random.choice([0,1], p=[1-prob_rep, prob_rep])

    # make data
    states[t] = state
    actions[t] = action

    # update
    state = 1 if action == 1 else min(5, state+1) # bounded [1,5]

# solve for theta = (mu,R) using Rust's NFP (MLE)
initial_guess = [-0.5, -2]
result = minimize(llh, initial_guess, method='BFGS')
mu_est, R_est = result.x

# solve for theta using Hotz and Miller's CCP approach with forward simulation
# estimate replacement probabilities using data
P_hat = np.array([np.mean(actions[states==a]) for a in age])

# forward simulation:
# conditional state transition matrices F0 and F1 (5x5)
F0 = np.zeros((5,5)) # keep
F1 = np.zeros((5,5)) # replace

for i in range(5):
    if i < 4:
        F0[i, i+1] = 1 # age + 1 if keep
    else:
        F0[i,i] = 1 # age = 5, max
    F1[i,0] = 1 # age = 1 if replace

# unconditional state transition matrix (5x5)
F_U = np.zeros((5,5))
for i in range(5):
    F_U[i,:] = P_hat[i] * F1[i,:] + (1-P_hat[i]) * F0[i,:]

# fwd sim using estimated replacement and unconditional transition probabilities
V0_hat, V1_hat = fwd_sim(P_hat, F_U, mu, R) # solve

# loglikelihood function for CCP
def llh_ccp(params):
    mu, R = params
    V0, V1 = fwd_sim(P_hat, F_U, mu, R)
    log_like = 0

    for t in range(T):
        a_t = states[t]
        i_t = actions[t]

        prob_rep_ccp = np.exp(V1[a_t-1]) / (np.exp(V0[a_t-1]) + np.exp(V1[a_t-1]))

        if i_t == 1:
            log_like += np.log(prob_rep_ccp)
        else:
            log_like += np.log(1 - prob_rep_ccp)

    return -log_like  # negative for minimize

result_ccp = minimize(llh_ccp, initial_guess, method='BFGS')
mu_ccp, R_ccp = result_ccp.x

# display results
print(V)
print("Indifference Condition at a_t=2:", eps_diff) # 0.1145
print("Probability of Replacing the Machine at a_t=2:", prob_replace) # 0.5286
print("Value function at state (a=4, eps_0=1, eps_1=1.5):", V_4) # -9.9001
print("Estimated mu (NFP):", mu_est) # -1.1625
print("Estimated R (NFP):", R_est) # -3.8230
print("Estimated mu (CCP approach):", mu_ccp) # -1.3884
print("Estimated R (CCP approach):", R_ccp) # 3.6680