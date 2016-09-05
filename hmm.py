from nltk.corpus import brown
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.cluster import KMeans
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
    self.gamma_tij = np.random.rand(self.T, self.N, self.N)
    self.gamma_ti  = np.random.rand(self.T, self.N)
    self.alpha = np.random.rand(T, N)  
    self.beta = np.random.rand(T, N)
    self.c = [0.0 for _ in range(self.T)]
    self.mapping = {}
    elements = string.ascii_uppercase.lower()
    for i in range(len(elements)):
      self.mapping[elements[i]] = i 
    self.mapping[' '] = 26

  def train(self):
    # Initializing some book keeping variables. 
    maxIters = 100
    iters = 0 
    oldLogProb = -np.inf
    logProb = 0 
    logLikilihood = []
    flag = False

    while iters < maxIters and logProb > oldLogProb:
      if flag:
        oldLogProb = logProb
      # alpha pass
      # compute alpha_0(i)
      self.c[0] = 0 
      for i in range(self.N):
        self.alpha[0, i] = self.pi[i] * self.B[i, self.mapping[self.O[0]]] 
        self.c[0] += self.alpha[0, i]    

      # scale alpha_0(i)  
      self.c[0] = 1.0/self.c[0]
      for i in range(self.N):
	self.alpha[0, i] = self.c[0] * self.alpha[0, i]

      # compute alpha_t(i)
      for t in range(1, self.T):
        self.c[t] = 0 
	for i in range(self.N):
	  self.alpha[t, i] = 0 

	  for j in range(self.N):
	    self.alpha[t, i] = self.alpha[t, i] + self.alpha[t-1, j] * self.A[j, i] 

	  self.alpha[t, i] = self.alpha[t, i] * self.B[i, self.mapping[self.O[t]]]     
	  self.c[t] += self.alpha[t, i] 

	# scale alpha_(t, i) 
	self.c[t] = 1.0/self.c[t]  
	for i in range(self.N):
	  self.alpha[t, i] = self.c[t] * self.alpha[t, i]   

      # setting beta_t-1 = 1 scaled by C_(T-1).
      for i in range(self.N):
	self.beta[self.T-1, i] = self.c[self.T - 1]

      # beta - pass
      for t in range(self.T - 2, 0, -1):
	for i in range(self.N):
	  self.beta[t, i] = 0 
	  for j in range(self.N):
	    self.beta[t, i] = self.beta[t, i] + self.A[i, j]*self.B[j,self.mapping[self.O[t+1]]]*self.beta[t+1,j]
	  self.beta[t, i] = self.c[t] * self.beta[t, i] 

      # compute gamma_(i, j) and gamma_t_i 
      for t in range(self.T - 1):
	denom = 0 
	for i in range(self.N):
	  for j in range(self.N):
	    denom = denom + (self.alpha[t, i] * self.A[i, j] * self.B[j, self.mapping[self.O[t+1]]] * self.beta[t+1, j])

	for i in range(self.N):
	  self.gamma_ti[t, i] = 0 
	  for j in range(self.N):
	    self.gamma_tij[t, i, j] = (self.alpha[t, i]*self.A[i, j] * self.B[j, self.mapping[self.O[t+1]]] * self.beta[t+1, j]) / denom 
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
	for j in range(self.M):
	  numer = 0.0 
	  denom = 0.0 
	  for t in range(self.T):
	    if self.mapping[self.O[t]] == j:
	      numer = numer + self.gamma_ti[t, i]
	    denom = denom + self.gamma_ti[t, i]
	  self.B[i, j] = numer/denom 

      # Compute log[P[O/lambda]]
      logProb = 0 
      for i in range(self.T):
	logProb = logProb + np.log(self.c[t])
      logProb = -logProb 
      logLikilihood.append(logProb)
      if iters >= 5:
        flag = True
      iters += 1

      # Print the current values of the lambda = (A, B, pi)
      print "\n\nIteration Number : ", iters 
      print self.A
      print self.B.T
      print self.pi
      print logProb

    return self.A, self.B, self.pi, logLikilihood

  def predict(self, O):
    # alpha pass
    # compute alpha_0(i)
    self.c[0] = 0 
    for i in range(self.N):
      self.alpha[0, i] = self.pi[i] * self.B[i, self.mapping[O[0]]] 
    self.c[0] += self.alpha[0, i]    

    # scale alpha_0(i)  
    self.c[0] = 1.0/self.c[0]
    for i in range(self.N):
      self.alpha[0, i] = self.c[0] * self.alpha[0, i]

    # compute alpha_t(i)
    for t in range(1, self.T):
      self.c[t] = 0 
      for i in range(self.N):
        self.alpha[t, i] = 0 

        for j in range(self.N):
	  self.alpha[t, i] = self.alpha[t, i] + self.alpha[t-1, j] * self.A[j, i] 

        self.alpha[t, i] = self.alpha[t, i] * self.B[i, self.mapping[O[t]]]     
        self.c[t] += self.alpha[t, i] 

      # scale alpha_(t, i) 
      self.c[t] = 1.0/self.c[t]  
      for i in range(self.N):
        self.alpha[t, i] = self.c[t] * self.alpha[t, i] 

    # logLikilihood 
    logLikilihood = 0.0
    for i in range(self.N):
      logLikilihood += self.alpha[self.T - 1, i] 

    return logLikilihood


if __name__ == "__main__":
  # Seeding the numpy random generator 
  np.random.seed(0)

  # Input an initial estimate of the perimeters A, B and prior
  B = np.vstack((np.random.dirichlet(np.ones(27),size=1), \
                 np.random.dirichlet(np.ones(27),size=1)))
  A = np.array([[0.47468, 0.52532],
                 [0.51656, 0.48344]])
  pi = np.array([0.51316, 0.48684])

  # Let's use the brown corpus for training 
  O = brown.words()
  O = ' '.join(O)
  O = O.lower()
  exclude = set(string.punctuation)
  O = ''.join(ch for ch in O if ch not in exclude)
  O = ''.join(i for i in O if not i.isdigit())
  O = O[:80000]

  hmm1 = hmm(pi.shape[0], 27, len(O), O, A, B, pi)
  A, B, pi, likilihood = hmm1.train()

  print "\n\n"
  pprint.pprint(pi)
  pprint.pprint(A)
  pprint.pprint(B.T)
  print "\n\n"
  print tabulate(zip([x[0] for x in sorted(hmm1.mapping.items(), key=lambda x:x[1])], B.T), \
                 headers=["Character", "Probability of being in each state"], tablefmt="pipe")

  x = [ i + 1 for i in range(len(likilihood))]
  plt.plot(x, likilihood)
  plt.xlabel("Number of iterations")
  plt.ylabel("Log likilihood")
  plt.show()
  plt.show()

  # Interpreting the results using clustering 
  B = B.T
  km = KMeans(n_clusters=2, \
              init="k-means++", \
              n_init=10, \
              max_iter=10, \
              tol=1e-04, \
              random_state=0)
  y_km = km.fit_predict(B)
  
  print tabulate(zip([x[0] for x in sorted(hmm1.mapping.items(), key=lambda x:x[1])], y_km),\
                 headers=["Character", "Cluster Assignmment"],tablefmt="pipe")

  # Plotting the alphabets to their clusters 
  plt.scatter(B[y_km==0, 0], 
              B[y_km==0, 1], 
              s=50, 
              c="lightgreen", 
              marker="s", 
              label="cluster 1")

  plt.scatter(B[y_km==1, 0], 
              B[y_km==1, 1], 
              s=50, 
              c="orange", 
              marker="o", 
              label="cluster 2")

  for i in range(B.shape[0]):
    plt.text(B[i, 0] + 0.001, B[i, 1] + 0.001, [x[0] for x in sorted(hmm1.mapping.items(), key=lambda x:x[1])][i])

  plt.legend()
  plt.grid()
  plt.show()

