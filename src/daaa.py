"""
This code is an implemenation of Deterministic Annealing Archetypal Analysis
It also includes a Pure Pixel Analysis variant

JLG 2025
"""

import numpy as np
import copy

class DAAA():
    def __init__(self, n_components=2, initial_regularization=1, time_constant=100, delta=0, epochs=10,
                record=False, mu=0):
        self.n_components = n_components
        self.initial_regularization = initial_regularization
        self.time_constant = time_constant
        self.delta=delta
        self.epochs=epochs
        self.PPA = False
        self.cos = False
        self.cos_per = 1
        self.use_weights=False
        self.verbose=True
        self.obj = 'L2'
        self.record = record
        self.mu = mu
        if self.record:
            self.obj_rec = []
       
        

    def init_tiered(self, max_rej=5):
        self.tiered = True
        self.tier = 0
        self.max_rej = 5
        self.rejections=0
    
    def _initialize(self, data):
        self.iteration = 0
        self.W = np.random.rand(data.shape[0], self.n_components)
        for i in range(self.n_components):
            #self.W[:,i] = np.mean(data, axis=1)
            self.W[:,i] = data[:,i]
        self.H = np.zeros((self.n_components, data.shape[1]))
        self.H[1,:] = 1
        self.normalize()
        #self.normalize_W()
        self.T = self.initial_regularization
        if self.use_weights:
            if self.obj=='L1':
                self.weights = 1/data.sum(axis=0)
            else:
                self.weights = 1/np.sqrt((data**2).sum(axis=0))
        else:
            self.weights = np.ones(data.shape[1])
    
        
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
        
    def append_record(self, data):
        if self.record:
            if self.obj=='L2':
                new_obj = objective(data,self.W,self.H,0,self.weights)
            elif self.obj=='Sp':
                new_obj = sp_objective(self.H)
            elif self.obj=='SpS':
                new_obj = (sp_objective(self.H) + size_objective(self.H))/2
            elif self.obj=='L1':
                new_obj = objectiveL1(data,self.W,self.H,0,self.weights)
            elif self.obj=='Size':
                new_obj = (size2_objective(nH))
            self.obj_rec.append([self.T, new_obj])
    
    def one_iterate(self, data):
        self.update_W_AA(data)
        self.update_all_abundances_BCD(data)
        
        self.append_record(data)
        
        self.iteration += 1
        self.T = self.initial_regularization*np.exp(-self.iteration / self.time_constant)
        if self.cos:
            self.T *= np.cos(2*np.pi*self.iteration/self.cos_per)

    def update_W(self, data):
        '''
        not used
        does not use weights
        '''
        self.W = np.multiply(self.W, np.multiply(data@self.H.T, 1/(self.W@self.H@self.H.T)))
        self.normalize_W()

    def update_W_AA(self, data):
        for i in range(self.n_components):
            
            err = data - self.W@self.H
            err_avg = err@((self.weights*self.H).T)
            dW = (data.T - self.W[:,i]).T 
            denoms = np.sum(dW**2, axis=0) * np.dot(self.weights*self.H[i,:],self.H[i,:])  
            impact = err_avg[:,i]@dW
            if self.PPA:
                beta_est = np.ones(len(err.T))
            else:
                beta_est = impact/(denoms+1e-16)
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

    def entropy(self):
        return np.sum(-self.H@np.log(self.H.T))
                                                         
    def train(self, data):
        for i in range(self.epochs*self.time_constant):
            self.one_iterate(data)
            if self.verbose:
                print(objective(data.T, self.W, self.H, self.T, self.weights))
            
    def save(self, name):
        if self.record:
            np.savez(name, H=self.H, W=self.W, R=np.array(self.obj_rec))
        else:
            np.savez(name, H=self.H, W=self.W)
        
        
    def update_all_abundances_BCD(self, data):
        self.H = update_all_abundances(data.T, self.W, self.H, self.T, self.mu)
        
    def SA_update_series(self, Y, gamma, reset_iter=True):
        n_steps = self.time_constant*self.epochs
        if reset_iter:
            self.iteration = 0 
        order = np.arange(len(self.H))
        while  self.iteration < n_steps:
            self.iteration+= 1
            np.random.shuffle(order)
            for j in order:
                if self.tiered:
                    prob_type = self.tier
                else:
                    prob_type = 2
                self.W, self.H, accept = SA_update(Y, self.W, self.H, gamma, j, self.T, self.weights, prob_type, self.obj)
                if self.tiered:
                    if accept==0:
                        self.rejections += 1
                    else:
                        self.rejections=0
                    if self.rejections > self.max_rej:
                        self.tier+=1
                        self.rejections=0
                        
                        
                self.append_record(Y)
            self.T = self.initial_regularization*np.exp(-self.iteration / self.time_constant)
            print("T is now ", self.T)
            
    def FCLS(self, X, gamma, tol=1e-4):
        self.H = np.ones((self.W.shape[1],len(X)))
        self.H /= self.W.shape[1]

        order = np.arange(len(self.H))
        diff = 1
        while diff > tol:
            np.random.shuffle(order)
            old_obj = objective(X, self.W, self.H, gamma, self.weights)
            for j in order:
                self.H = FCLS_onestep(X, self.W, self.H, j, gamma)
            new_obj = objective(X, self.W, self.H, gamma, self.weights)
            diff = (1 - new_obj/old_obj)
            print("update size is ", new_obj, old_obj)
        return H
              
        
def update_one_endmember_abundance(Y, W, H, ID_n, gamma, mean_unmixed=False):
    # Define the converse spectrum where it is not well-defined
    L = len(H)
    EPS = 0.01
    pp = H[ID_n]==1 #pure pixels
    if mean_unmixed:
        H[:,pp] = EPS/(L-1)
        H[ID_n,pp] = 1-EPS
        pp[pp] = False
    
    converse_spectra = np.zeros_like(Y)
    converse_abundances_sum = np.zeros(H.shape[1])
    converse_abundances_sqsum = np.zeros(H.shape[1])
    for i in range(len(H)):
        if i != ID_n:
            converse_abundances_sum += H[i]
            converse_abundances_sqsum += H[i]**2
            converse_spectra += np.outer(H[i], W[:,i])
    converse_spectra = converse_spectra.T
    converse_spectra[:,~pp] /= converse_abundances_sum[~pp]
    converse_spectra = converse_spectra.T
    stability_parameter = np.sum((converse_spectra[~pp]-W[:,ID_n])**2, axis=-1)
    stability_parameter += gamma*(1+converse_abundances_sqsum[~pp]/converse_abundances_sum[~pp]**2)
    stable = stability_parameter > 0
    lambdas = np.zeros_like(converse_abundances_sum)
    nums = np.sum((Y[~pp][stable]-converse_spectra[~pp][stable])*(W[:,ID_n]-converse_spectra[~pp][stable]),
                  axis=-1)
    nums += gamma*converse_abundances_sqsum[~pp][stable]/converse_abundances_sum[~pp][stable]**2
    full_stable = ~pp
    full_stable[~pp] = stable
    lambdas[full_stable] = nums/(stability_parameter[stable])
    #print(stable.sum())
    if np.sum(~stable)>0:
        full_instable = ~pp
        full_instable[~pp] =  ~stable
        err_other = np.sum((Y[full_instable]-converse_spectra[full_instable])**2, axis=-1) \
                + gamma*converse_abundances_sqsum[full_instable]
        err_id = np.sum((Y[full_instable]-W[:,ID_n])**2, axis=-1) +  gamma
        
        lambdas[full_instable]= np.argmin(np.array([err_other, err_id]), axis=0)
    lambdas[pp] = 1
    lambdas[lambdas<0]=0
    lambdas[lambdas>1]=1
    return lambdas, converse_abundances_sum


def gentle_update_abundance(Y, W, H, i, mu):
    cas = 1-H[i] # here we assume the 1-v-rest orientation 
    
    m_n, m_d = calc_m_adjustments(i, H, W)
    b_n, b_d, a1 = calc_err_grad(i, H, W, Y.T)

    lam = np.minimum(np.maximum(0,(b_n + mu*m_n)/(b_d + mu*m_d)),1)

    return lam, a1
    

def update_all_abundances(Y, W, H, gamma, mu=0):
    ems = np.arange(len(H))
    np.random.shuffle(ems)
    for i in ems:
        if mu == 0:
            lam, cas = update_one_endmember_abundance(Y, W, H, i, gamma)
            H[:,cas>0] /= cas[cas>0]
            H *= (1-lam)
            H[i] = lam
        else: #only set up to use either gamma or mu, assumes gamma is 0 if mu is non-zero
            #print(i, objective(Y, W, H, 0, 1),'x')
            lam, a1 = gentle_update_abundance(Y, W, H, i, mu)
            vals = np.zeros(len(H), dtype=bool)
            vals[i] = True
            #print(vals)
            H[vals] = lam
            H[~vals] = (1-lam)*a1
            #print(i, objective(Y, W, H, 0, 1))
            #print(H.dtype)
            #print(a1.shape, lam.shape, vals.sum())
        
       
        #print((gamma*np.sum((H)**2))/len(Y))
    
    return H

    
def SA_update(Y,W,H,gamma, ID_n, T, weights, prob_type=2, obj='L2'):
    L = len(Y)
    s = len(H)
    if obj=='L2':
        original_obj = objective(Y,W,H,gamma,weights)
    elif obj=='Sp':
        original_obj = sp_objective(H)
    elif obj=='SpS':
        original_obj = (sp_objective(H) + size_objective(H))/2
    elif obj=='L1':
        original_obj = objectiveL1(Y,W,H,gamma,weights)
    elif obj=='Size':
        original_obj = (size2_objective(H))
    if prob_type==1:
        new_pix_num = np.random.choice(L, 1, p=(H[ID_n]>0)/(H[ID_n]>0).sum())
    elif prob_type==0:
        new_pix_num = np.random.choice(L, 1)
    else:
        new_pix_num = np.random.choice(L, 1, p=H[ID_n]/H[ID_n].sum())
    nW = copy.deepcopy(W)
    nW[:,ID_n] = Y[new_pix_num]
    #print(nW)
    nH= copy.deepcopy(H)
    EPS = 0.01
    pp = nH[ID_n]==1 #pure pixels
    nH[:,pp] = EPS/(L-1)
    nH[ID_n,pp] = 1-EPS
    lam, cas = update_one_endmember_abundance(Y, nW, nH, ID_n, gamma)
    nH = nH*(1-lam)
    nH[:, cas>0] /= cas[cas>0]
    nH[ID_n] = 0
    nH[ID_n] = 1-np.sum(nH, axis=0)
    nH[nH<0]=0
    nH[nH>1]=1
    
    #nH = daaa.update_all_abundances(Y, nW, nH, gamma)
    
    #lam, cas = daaa.update_one_endmember_abundance(Y, nW, nH, ID_n, gamma)
    #nH = nH*(1-lam)
    #nH[:, cas>0] /= cas[cas>0]
    #nH[ID_n] = lam
    
    r = np.random.rand()
    if obj=='L2':
        new_obj = objective(Y, nW, nH, gamma,weights)
    elif obj=='Sp':
        new_obj = sp_objective(nH)
    elif obj=='SpS':
        new_obj = (sp_objective(nH) + size_objective(nH))/2
    elif obj=='L1':
        new_obj = objectiveL1(Y,nW,nH,gamma,weights)
    elif obj=='Size':
        new_obj = (size2_objective(nH))
    dE = new_obj-original_obj
    #print(new_obj, original_obj, np.exp(-dE/T), r)
    accept = r < np.exp(-dE/T)
    if accept:
        print(new_obj, original_obj, np.exp(-dE/T), r)
    else:
        print('*',new_obj, original_obj, np.exp(-dE/T), r)
    if accept:
        return nW, nH, accept
    else:
        return W, H, accept

def objective(Y, W, H, gamma, weights):
    return((np.sum(weights*(Y.T-W@H)**2)+gamma*np.sum((H)**2))/len(Y))

def objectiveL1(Y, W, H, gamma, weights):
    return((np.sum(weights*np.abs(Y.T-W@H)))/len(Y))

def sp_objective(H):
    return 1-np.sum(H**2)/H.shape[1]

def size_objective(H):
    N = H.shape[1]
    P = H.shape[0]
    #print(N,P,np.sum(H, axis=1))
    return P/(N-N/P)**2 * np.sum((np.sum(H, axis=1)-N/P)**2)

def size2_objective(H):
    N = H.shape[1]
    P = H.shape[0]
    #print(N,P,np.sum(H, axis=1))
    return 1/(N-N/P)**2 * np.sum((np.sum(H**2, axis=1)-N/P)**2)

def FCLS_onestep(X, W, H, ID_n, gamma):
    L = len(H)
    nH= copy.deepcopy(H)
    #EPS = 0.01
    #pp = nH[ID_n]==1 #pure pixels
    #nH[:,pp] = EPS/(L-1)
    #nH[ID_n,pp] = 1-EPS
    lam, cas = update_one_endmember_abundance(X, W, nH, ID_n, gamma)
    nH = nH*(1-lam)
    nH[:, cas>0] /= cas[cas>0]
    nH[ID_n] = 0
    nH[ID_n] = 1-np.sum(nH, axis=0)
    return nH

def calc_oppo_L_a1(endmember, S, spectra):
    oppo_list = [i for i in range(len(S)) if i != endmember]
    #print(oppo_list, endmember)
    L = S.shape[-1]
    a1 = np.array([S[i]+1e-8 for i in oppo_list]) 
    a1 /= np.sum(a1, axis=0)
    return oppo_list, L, a1

def calc_m_adjustments(endmember, S, spectra):
    oppo_list, _, a1 = calc_oppo_L_a1(endmember, S, spectra)
    a_x = lambda x: (np.sum(a1**x, axis=0))
    a2 = a_x(2)
    a3 = a_x(3)

    mx_curve = 12*(1-a3)-6-6*a2+12*a3 #numbers come from reg.
    y0 = S[endmember]
    lin =  (6*(1-a3)*y0**2-6*(1+a2-2*a3)*y0+6*(a2-a3))

    m_numerator = mx_curve*y0-lin
    m_denominator = mx_curve

    return m_numerator, m_denominator

def calc_err_grad(endmember, S, spectra, data):
    oppo_list, L, a1 = calc_oppo_L_a1(endmember, S, spectra)
    conv_spec = np.sum([np.outer(a1[i], spectra[:,oppo_list[i]]) for i in range(len(oppo_list))], axis=0).astype(np.float128)

    basic_numerator = ((data.T-conv_spec).reshape(L,1,-1)@((spectra[:,endmember])-conv_spec).reshape(L,-1,1)).reshape(L)
    delta = (spectra[:,endmember]-conv_spec).astype(np.float128)
    basic_denominator = (delta.reshape(L,1,-1)@delta.reshape(L,-1,1)).reshape(L)
    
    return basic_numerator, basic_denominator, a1

def make_mu_adj(basic_numerator, basic_denominator, m_numerator, m_denominator):
    return lambda x: np.maximum((basic_numerator + x * m_numerator)/(basic_denominator + x * m_denominator),0)
    