import numpy as np
import scipy.sparse as sp
from random import sample
from scipy.linalg import pinv, inv

class VSRP():
    def __init__(self, n_bands=-1):
        if n_bands>0:
            self.n_bands = n_bands
        else:
            self.n_bands = n_bands
            
    def fit(self, img):
        if self.n_bands == -1:
            self.n_bands = img.shape[-1]
        self.mean = np.mean(img, axis=0)
        self.proj = vsrp(img, r=self.n_bands).T
        self.proj = np.array(self.proj.todense())

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

    def save(self, file_name):
        np.savez(file_name, proj=self.proj, mean=self.mean)
    
    def load(self, file_name):
        loaded = np.load(file_name)
        self.proj = loaded['proj']
        self.n_bands = self.proj.shape[-1]
        self.mean = loaded['mean']
        

def vsrp(img, r=0, density="auto"):
    
    '''              Very sparse random projection
    ---------------------------------------------------------------
    Inputs: img - size(noPixel x noBands) 
            r   - the reduced dimension (r<<noBands)
    Outputs: redImg - size(noPixels x r): redImg = 1/sqrt(r) x img x projMatrix^T
    ---------------------------------------------------------------
    The components of the random projection matrix (denoted projMatrix) are formed as follows:
    
      - -sqrt(s)    with probability 1 / 2s
      -  0          with probability 1 - 1 / s
      - +sqrt(s)    with probability 1 / 2s
      
    where s = sqrt(noBands)
    
    '''
    noFeatures = img.shape[-1]             # noFeatures = noBands
    rng = np.random.default_rng(0)
    indices = []
    offset = 0
    indptr = [offset]
    
    if r == 0:
        r = img.shape[-1]
    
    #Check density
    if density == "auto":
        density = 1 / np.sqrt(noFeatures)   # other choise to try s = n/log(n) (very sparse)
                                            # Achlioptas used s = 1 or s = 3 (sparse)
    elif density<=0 or density > 1:
        print("Value must be between (0,1]")   
    
    #Create the random projection matrix full of +/- 1 and 0
    featureIndx = range(noFeatures)
    
    for _ in range(r):
        # find the indices of the non-zero components for row i
        n_nonzero_i = rng.binomial(noFeatures, density)
        indices_i = sample(featureIndx, n_nonzero_i)
        indices.append(indices_i)
        offset += n_nonzero_i
        indptr.append(offset)

    indices = np.concatenate(indices)
    # Among non zero components the probability of the sign is 50%/50%
    data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

    # build the Compressed Spare Row matrix structure by concatenating the rows
    projMatrix = sp.csr_matrix(
            (data, indices, indptr), shape=(r, noFeatures))
    
    #redImg = np.sqrt(1 /density)/ np.sqrt(r) * img @ projMatrix.T  
    
    return np.sqrt(1 /density)/ np.sqrt(r)*projMatrix.T
