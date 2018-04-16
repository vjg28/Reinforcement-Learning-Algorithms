import numpy as np

def one_step_lookahead(environment, state, V, discount_factor):
    """
    helper function to calculate the value function
    
    """
    #Creating a vector of dimensionally same size as the number of actions
    action_values=np.zeros(environment.nA)
    
    for action in range(environment.nA):
        
        for probability, next_state, reward, terminated in environment.P[state][action]: #policy
            action_values[action] +=  probability * (reward + discount_factor * V[next_state])
            
    return action_values    
    
def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iteration=1e9):
    """
    evaluate a policy given a deterministic environment
    
    1)policy : Matrix of size nS*nA. Each cell reprents the probability of 
               taking an action in a particular state
    2)Environment : openAI environment object
    3)discount_factor:
    4)theta: Convergence factor. If the change in value function for all 
             states is below theta, we are done.
    5)max_iteration: To avoid infinite looping.
    
    Returns:
    1)V:The optimum value estimate for the given policy 
    """
    
    evaluation_iteration = 1 # to record the number of iteration
    V=np.zeros(environment.nS)
    
    for i in range(int(max_iteration)):
        delta = 0  #for early stopping
        
        for state in range(environment.nS):
            v=0
            
            for action, action_probability in enumerate(policy[state]):
                
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    #print(state, next_state,state_probability)
                    v+= action_probability * state_probability * (reward + discount_factor *V[next_state])
        
            delta= max(delta, abs(V[state]-v)) #looks like a mistake here.Check
            V[state]=v

        evaluation_iteration +=1
        
        if(delta < theta):
            print('Policy evaluated in %d iteration' % evaluation_iteration)
            return V
            
def policy_iteration(environment, discount_factor=1.0, max_iteration=1e9):
    
    """
    In this function, we would take a random policy and evaluate the optimum 
    value function of the policy ,act greedily on the policy and work for the
    new better policy.
    """
    
    policy=np.ones((environment.nS, environment.nA))/environment.nA
    
    evaluated_policies =1
    
    for i in range(int(max_iteration)):
        
        stable_policy= True
        V=policy_evaluation(policy,environment, discount_factor=discount_factor,max_iteration=max_iteration)
        
        for state in range(environment.nS):
            
            current_action=np.argmax(policy[state])    #error here what if elements are same?
            action_values=one_step_lookahead(environment, state,V ,discount_factor=discount_factor)
            best_action = np.argmax(action_values)
            
            if(current_action != best_action):
                stable_policy =False
                
            policy[state]=np.eye(environment.nA)[best_action]
            
        evaluated_policies +=1
        
        if(stable_policy):
            print('Evaluated %d policies.' % evaluated_policies)
            return policy, V

def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iteration=1e9):
    
    V=np.zeros(environment.nS)
    
    for i in range(int(max_iteration)):
        
        delta=0
        
        for state in range(environment.nS):
            
            action_values=one_step_lookahead(environment, state, V, discount_factor)
            best_action_value=np.max(action_values)
            delta=max(delta,abs(V[state]-best_action_value))
            V[state]=best_action_value
            
        if(delta <theta):
            print('Value iteration converged at iteration #%d' % i)
            break
    
    policy= np.zeros((environment.nS, environment.nA))
    
    for state in  range(environment.nS):
        
        action_values= one_step_lookahead(environment, state, V, discount_factor)
        best_action = np.argmax(action_values)
        policy[state][best_action]=1.0
        
    return policy, V
            