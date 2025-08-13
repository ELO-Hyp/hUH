"""
This code is an implemenation of Deterministic Annealing Archetypal Analysis
It also includes a Pure Pixel Analysis variant

JLG 2025
"""

import numpy as np

class DAAA():
    def __init__(self, n_components=2, initial_regularization=1, time_constant=100, delta=0, epochs=10):
        self.n_components = n_components
        self.initial_regularization = initial_regularization
        self.time_constant = time_constant
        self.delta=delta
        self.epochs=epochs
        self.PPA = False

    def _initialize(self, data):
        self.iteration = 0
        self.W = np.random.rand(data.shape[0], self.n_components)
        for i in range(self.n_components):
            self.W[:,i] = np.mean(data, axis=1)
        self.H = (1/self.n_components)*(1+np.random.rand(self.n_components, data.shape[1]))
        self.normalize()
        #self.normalize_W()
        self.T = self.initial_regularization
        
    def normalize(self):
        '''
        just normalize the data first
        '''
        self.H /= np.sum(self.H, axis = 0)

    def normalize_W(self):
        '''
        just normalize the data first
        '''
        self.W = np.maximum(self.W, 1e-5)
        self.W = (self.W/np.sqrt(np.sum(self.W**2, axis=0)))    
        
    def one_iterate(self, data):
        self.update_W_AA(data)
        self.update_H(data)
        self.iteration += 1
        self.T = self.initial_regularization*np.exp(-self.iteration / self.time_constant)

    def update_W(self, data):
        '''
        not used
        '''
        self.W = np.multiply(self.W, np.multiply(data@self.H.T, 1/(self.W@self.H@self.H.T)))
        self.normalize_W()

    def update_W_AA(self, data):
        for i in range(self.n_components):
            
            err = data - self.W@self.H
            err_avg = err@self.H.T
            dW = (data.T - self.W[:,i]).T 
            denoms = np.sum(dW**2, axis=0) * np.dot(self.H[i,:],self.H[i,:])  
            impact = err_avg[:,i]@dW
            if self.PPA:
                beta_est = np.ones(len(err.T))
            else:
                beta_est = impact/denoms
                beta_est[beta_est>1] = 0
                beta_est[beta_est<0] = 0
            
            #print(beta_est.max())
            change = -2*beta_est*impact + beta_est**2 * denoms
            #print(np.argmin(change), np.argmax(beta_est))
            changeids = np.argsort(change)
            beta_id = changeids[0]
            if self.PPA:
                j = 0
                unaccepted = True
                while unaccepted:
                    diff = np.sum((data[:,beta_id] - self.W.T)**2, axis=-1)
                    
                    if np.min(diff)==0:
                        j+= 1
                        beta_id = changeids[j]
                        #print(j)
                    else:
                        unaccepted=False
                #print(beta_id, min(diff), np.argmin(diff), beta_est[beta_id])
            beta = beta_est[beta_id]
            
            #print(beta_id, beta_id.dtype)
            self.W[:,i] = (1-beta)*self.W[:,i] + beta*data[:,beta_id]
    
    def update_H(self, data):
        Data_a = np.append(data, self.delta * np.ones((1, len(data.T))), axis=0)
        Wa = np.append(self.W, self.delta* np.ones((1,self.n_components)), axis =0)
        self.H = np.multiply(self.H, np.multiply(Wa.T@Data_a - self.T*(np.log(self.H)),
                                                 1/(Wa.T@Wa@self.H + self.T)))
        self.H = np.maximum(self.H, 1e-5)
        #self.normalize()T
4
    def entropy(self):
        return np.sum(-self.H@np.log(self.H.T))
                                                         
    def train(self, data):
        for i in range(self.epochs*self.time_constant):
            self.one_iterate(data)
            
    def save(self, name):
        np.savez(name, H=self.H, W=self.W)
                                                         
                                                        

    