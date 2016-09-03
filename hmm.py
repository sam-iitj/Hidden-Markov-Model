from nltk.corpus import brown
import numpy as np 
import nltk 
import string
import pprint

class hmm:

  def __init__(self, N, M, T, O, A, B, pi):
    """
    Initialize various parameters of the HMM model. 
    A : { N X N } - transition matrix a_i_j is probaility of transition from state i to state j. 
    B : { N X M } - probability matrix of producting symbol m in state i, identified by b_i(O_m). 
    pi: { 1 X N } - initial probability of being in each state.
    alpha : { T X N } - values to compute at alpha step.  
    M : size of vocabulary. 
    N : size of state space. 
    T : length of the input sequence O 
    O : Observations 
    """
    self.N = N
    self.M = M
    self.T = T
    self.O = O  
    self.A = A
    self.B = B
    self.pi = pi
    self.gamma_tij = np.zeros((self.T, self.N, self.N))
    self.gamma_ti  = np.zeros((self.T, self.N))
    self.alpha = np.zeros((T, N))  
    self.beta = np.zeros((T, N))

  def train(self):
    # alpha pass
    # compute alpha_0(i)
    c = [0.0 for _ in range(self.T)]
    for i in range(self.N):
      self.alpha[0, i] = self.pi[i] * self.B[i, ord(self.O[0])-65] 
      c[0] = c[0] + self.alpha[0, i]    

    # scale alpha_0(i)  
    c[0] = 1.0/c[0]
    for i in range(self.N):
      self.alpha[0, i] = c[0] * self.alpha[0, i]

    # compute alpha_t(i)
    for t in range(1, self.T):
      for i in range(self.N):
        self.alpha[t, i] = 0 

        for j in range(self.N):
          self.alpha[t, i] = self.alpha[t, i] + self.alpha[t-1, j] * self.A[j, i] 

        self.alpha[t, i] = self.alpha[t, i] * self.B[i, ord(self.O[t])-65]     
        c[t] = c[t] + self.alpha[t, i] 

      # scale alpha_(t, i) 
      c[t] = 1.0/c[t]  
      for i in range(self.N):
        self.alpha[t, i] = c[t] * self.alpha[t, i]   

    # setting beta_t-1 = 1 scaled by C_(T-1).
    for i in range(self.N):
      self.beta[self.T-1, i] = c[self.T - 1]

    # beta - pass
    for t in range(self.T - 2, 0, -1):
      for i in range(self.N):
        self.beta[t, i] = 0 
        for j in range(self.N):
          self.beta[t, i] = self.beta[t, i] + self.A[i, j]*self.B[j,ord(self.O[t+1])-65]*self.beta[t+1,j]
        self.beta[t, i] = c[t] * self.beta[t, i] 

    # compute gamma_(i, j) and gamma_t_i 
    for t in range(self.T - 1):
      denom = 0 
      for i in range(self.N):
        for j in range(self.N):
          denom = denom + (self.alpha[t, i] * self.A[i, j] * self.B[j, ord(self.O[t+1]) - 65] * self.beta[t+1, j])

      for i in range(self.N):
        self.gamma_ti[t, i] = 0 
        for j in range(self.N):
          self.gamma_tij[t, i, j] = (self.alpha[t, i]*self.A[i, j] * self.B[j, ord(self.O[t+1])-65] * self.beta[t+1, j]) / denom 
          self.gamma_ti[t, i] = self.gamma_ti[t, i] + self.gamma_tij[t, i, j]

    denom = 0.0 
    for i in range(self.N):
      denom = denom + self.alpha[self.T - 1, i]
    for i in range(self.N):
      self.gamma_ti[self.T - 1, i] = self.alpha[self.T - 1, i]/denom

    # Reestimate A, B and pi
    # Reestimate pi 
    for i in range(self.N):
      self.pi[i] = self.gamma_ti[0, i] 
    
    # Restimate A 
    for i in range(self.N):
      for j in range(self.N):
        numer = 0.0 
        denom = 0.0 
        for t in range(self.T - 1):
          numer = numer + self.gamma_tij[t, i, j] 
          denom = denom + self.gamma_ti[t, i] 
        self.A[i, j] = numer/denom

    # Reestimate B 
    for i in range(self.N):
      for j in range(self.N):
        numer = 0.0 
        denom = 0.0 
        for t in range(self.T - 1):
          if ord(self.O[t]) - 65 == j:
            numer = numer + self.gamma_ti[t, i]
          denom = denom + self.gamma_ti[t, i]
        self.B[i, j] = numer/denom 

    # Compute log[P[O/lambda]]
    logProb = 0 
    for i in range(self.T):
      logProb = logProb + np.log(c[t])
    logProb = -logProb 
    return logProb

if __name__ == "__main__":
  # Input an initial estimate of the perimeters A, B and prior
  first = np.array([1.0/26.0 for _ in range(26)])
  second = np.array([1.0/26.0 for _ in range(26)])
  B = np.vstack((first, second))
  A = np.array([[0.47468, 0.52532],[0.51656,0.48344]])
  pi = np.array([0.51316, 0.48684])

  # Let's use the brown corpus for training 
  O = brown.words()
  O = ''.join(O)
  O = O.upper()
  exclude = set(string.punctuation)
  O = ''.join(ch for ch in O if ch not in exclude)
  O = ''.join(i for i in O if not i.isdigit())

  hmm1 = hmm(2, 26, len(O), O, A, B, pi)

  maxIters = 10
  iters = 0 
  oldLogProb = -np.inf
  output = 0.0

  # Iterating the algo for 10 iterations
  while iters < maxIters:
    if output > oldLogProb:
      oldLogProb = output
    output = hmm1.train()
    print "Ouput : " , output 
    iters += 1 

  pprint.pprint(hmm1.pi)
  pprint.pprint(hmm1.A)
  pprint.pprint(hmm1.B)
