import numpy as np
from scipy import linalg
from numpy.linalg import pinv

def defIca(X, g, gprime, r, maxit, tol):
    
    """Deflationary FastICA  """
    
    #Initialize w
    w_init = np.random.normal(size=(r, X.shape[0]))
    W = np.zeros((r,X.shape[0]), dtype=float)

    # j is the index of the extracted component
    for j in range(r):
        w = w_init[j, :].copy()
        w /= np.sqrt((w**2).sum())

        n_iterations = 0
        # we set lim to tol+1 to be sure to enter at least once in next while
        crit = tol + 1 
        while ((crit > tol) & (n_iterations < (maxit-1))):
            wtx = np.dot(w.T, X)
            gwtx = g(wtx,)
            g_wtx = gprime(wtx)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
            
            # Gram-Schmidt-like decorrelation
            t = np.zeros_like(w1)
            for i in range(j):
                t = t + np.dot(w1, W[i]) * W[i]
                w1 -= t
           
            # Normalize
            w1 /= np.sqrt((w1**2).sum())
            # Check critrium
            crit = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            n_iterations = n_iterations + 1
            
        W[j, :] = w
        #print(X.shape, w.shape)
        X -= (np.outer(X.T@w, w)).T
    return W

def symDecorr(W):
    """ Symmetric decorrelation """
    #print(W)
    K = np.dot(W, W.T)
    #print(K)
    s, u = linalg.eigh(K) 
    u, W = [np.asmatrix(e) for e in (u, W)]
    #added abs value to make the code work, not sure if it is mathematically valid
    W = (u * np.diag(1.0/np.sqrt(np.abs(s))) * u.T) * W  # W = (W * W.T) ^{-1/2} * W
    return np.asarray(W)

def parIca(X, g, gprime, r, maxit,tol):
    """Parallel FastICA  """
    
    #Initialize w
    n, p = X.shape
    print(r, n)
    w_init = np.random.normal(size=(r, n))
    # Decorelate the initial guess:
    W = symDecorr(w_init)

    # we set lim to tol+1 to be sure to enter at least once in next while
    crit = tol + 1 
    it = 0
    while ((crit > tol) and (it < (maxit-1))):
        wtx = np.dot(W, X)
        gwtx = g(wtx)
        g_wtx = gprime(wtx)
        W1 = np.dot(gwtx, X.T)/float(p) - np.dot(np.diag(g_wtx.mean(axis=1)), W)
 
        W1 = symDecorr(W1)
        
        crit = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        it += 1

    return W


class FastICA():
    def __init__(self, n_bands=-1,
                 fun = 'exp', redPCA = 1,
                 alg = 1, maxit=200,
                 tol=1e-04):
        self.n_bands = n_bands
        self.fun = fun
        self.redPCA = 1
        self.alg = alg
        self.maxit = maxit
        self.tol = tol
    def fit(self, img, *args):
        if self.n_bands == -1:
            self.n_bands = img.shape[-1]

        self.mean = np.mean(img, axis=0)
        self.proj = fastICA(img-self.mean, r=self.n_bands,
                            fun=self.fun, redPCA=self.redPCA,
                            alg=self.alg, maxit=self.maxit,
                            tol=self.tol).transpose()
    def set_r(self, r):
        if r == -1:
            r = self.n_bands
        return r

    def transform(self, img, r=-1):
        r = self.set_r(r)
        if r== self.n_bands:
            #print(self.proj.shape, (img-self.mean).shape)
            return self.proj@(img-self.mean).transpose()
        else:
            proj = self.proj[:r]
            return proj @ (img - self.mean).transpose()

    def inverse_transform(self, tr_img, r=-1):
        r = self.set_r(r)
        try:
            inv = pinv(self.proj[:r])
            return (inv @ tr_img).transpose() + self.mean
            # the following is for testing
        except np.linalg.LinAlgError:
            print("SVD failed, not inverting")
            inv = np.zeros(self.proj[:r].shape)
            return (inv @ tr_img).transpose()

    def endecode(self, img, r=-1):
        r = self.set_r(r)
        return self.inverse_transform(self.transform(img, r), r)

    def save(self, file_name):
        np.savez(file_name, proj=self.proj,
                 mean=self.mean,
                 parameters=np.array([
                     self.fun,
                     self.redPCA,
                     self.alg,
                     self.maxit,
                     self.tol
                 ]))

    def trained_weights(self):
        props = {
            'projection': self.proj,
            'mean': self.mean
         }
        return props

    def load_from_props(self, projection, mean):
        #meta parameters already defined, should only be used when they are known
        self.proj = projection
        self.n_bands = self.proj.shape[-1]
        self.mean = mean

    def load(self, file_name):
        loaded = np.load(file_name)
        self.proj = loaded['proj']
        self.n_bands = self.proj.shape[-1]
        self.mean = loaded['mean']
        self.fun = loaded['parameters'][0]
        self.redPCA = loaded['parameters'][1]
        self.alg = loaded['parameters'][2]
        self.maxit = loaded['parameters'][3]
        self.tol = loaded['parameters'][4]

def fastICA(img, r = 0, fun = 'exp', redPCA = 1, alg = 1, maxit=200, tol=1e-04):
    """ FastICA
    --------------------------------------------------------------
    Inputs: img - size(noPixels x noBands) 
            r   - the reduced dimension (r<<noBands)
            fun - {'exp','logcosh','cube'} functions that aproximate negentropy
            redPCA - 0: doesn't reduce the dimensions during whitening
                   - 1: reduceS the dimension noBands to r during whitening
            alg - 0: deflation strategy for solving ICA, meaning find one w at a time
                - 1: parallel fastica, i.e find all w at the same time
            maxit - maximum number of iteration 
            tol   - precision of the solution
    NOTE: W is the projection matrix we are looking for. w is a column of W.
    """
    noPixels, noBands = img.shape
    if r == 0:
        r = noBands
    
    # Chose function g and g':
    if fun == 'logcosh':
        def g(x, alpha = 1.0):
            return np.tanh(alpha * x)
        def gprime(x, alpha = 1.0):
            return alpha * (1 - (np.tanh(alpha * x))**2)
        
    elif fun == 'exp':
        def g(x):
            return x * np.exp(-(x**2)/2)
        def gprime(x):
            return (1 - x**2) * np.exp(-(x**2)/2)
        
    elif fun == 'cube':
        def g(x):
            return x**3
        def gprime(x):
            return 3*x**2
    
    # Centering
    img = img -np.mean(img, axis =0) 
    
    # Whitening using PCA : imgW@imgW = I_b
    u, d, _ = np.linalg.svd(img.T, full_matrices=False)
    K = (u / d).T  # see (6.33) p.140
    del u, d, _
    # Use PCA reduction
    if redPCA == 0:                    #Obs: seams like without the PCA reduction,
        imgW = np.dot(K, img.T)        #     the components don't look "nice" on the whitened data
    else: 
        imgW = np.dot(K[:r], img.T)
        K = K[:r]
        
    imgW *= np.sqrt(noPixels)
    if alg == 0:
        W = defIca(imgW, g, gprime, r, maxit,tol)
    else: 
        W = parIca(imgW, g, gprime, r, maxit,tol)
    
    #redImg = np.dot(img, np.dot(W, K).T)
    
    return (W@K).T #SWredImg
