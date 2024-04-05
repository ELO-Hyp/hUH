import numpy as np
from scipy.linalg import pinv

def pca(img, r=0):
    '''              Principal component analysis (PCA)
    ---------------------------------------------------------------
    Inputs: img - size(noPixel x noBands) 
            r   - the reduced dimension (r<<noBands)
    Outputs: redImg - size(noPixels x r)
    ---------------------------------------------------------------    
    '''
    if r == 0:
        r = img.shape[-1]
    
    #Centering: remove mean
    img = img -np.mean(img, axis =0)
    
    U, S, V  = np.linalg.svd(img, full_matrices = False)  # the dimensions for U[noPixel,k] & Vt[k,noBands],
                                                           # where k=min(noPixel, noBands)
    # redImg = Img*V = U*S*Vt*V = U*S
    #redImg = U[:,:r]*S[:r]  
    return V[:r] #redImg




def nysPCA(img, r=0, me =0):
    ''' Nystrom approximation of PCA
    ----------------------------------------------
    Inputs: img - size(noPixel x noBands) 
            r   - the reduced dimension (r<<noBands)
            me - 0: numerical stable version, but needs to memorize a matrix of dimension noPixel x r
                 1: memory efficient (works with matrix r x r), but need to compute covariance matrix 
    Outputs: redImg - size(noPixels x r)
    ---------------------------------------------------------------   
    '''
    noBands = img.shape[-1]
    if r == 0:
        r = noBands
    
    #Centering: remove mean
    img = img -np.mean(img, axis =0) 
    
    #Column sampling
    colIdx = np.random.choice(noBands, r, replace=False)
    # colIdx = np.sort(colIdx)   # not sure if necessary to be in order????
    
    if me ==0:                 #more numerical stable
        U,S, _ = np.linalg.svd(img[:,colIdx],full_matrices = False)
        Vnys = img.T @ U @ np.linalg.inv(np.diag(S))
       
    else:                     #more memory effcient
        covImg = 1/img.shape[0] * img[:,colIdx].T @img[:,colIdx] # dim r x r
        _,S,Vt = np.linalg.svd(covImg)
        Vnys =  img.T@ img[:,colIdx]@ Vt.T@ np.linalg.inv(np.diag(S))
        
    #redImg = np.sqrt(r/noBands)* img @ Vnys
    return np.sqrt(r/noBands)*Vnys#redImg




def csPCA(img, r=0):
    ''' Column sampling approximation of PCA
    ----------------------------------------------
    Inputs: img - size(noPixel x noBands) 
            r   - the reduced dimension (r<<noBands)
    Outputs: redImg - size(noPixels x r)
    ---------------------------------------------------------------   
    '''
    noBands = img.shape[-1]
    if r == 0:
        r = noBands
        
    #Centering: remove mean
    img = img -np.mean(img, axis =0)
    
    #Column sampling
    colIdx = np.random.choice(noBands, r, replace=False)
    img1 = 1/img.shape[0] * img.T@ img[:,colIdx]
    
    Vcs,_,_ = np.linalg.svd(img1)
    #redImg = img@Vcs[:,:r]
        
    return Vcs#redImg

alg_types = {0:pca, 1:csPCA, 2:nysPCA, 3:nysPCA}
# 0, 2 - SVD on full matrix, 1,3 SVD on covariance matrix

class PCA():
    def __init__(self, n_bands=0, alg_type=0):
        self.n_bands = n_bands
        self.alg_type = alg_type
        if (alg_type // 2) > 0:
            if (alg_type % 2 ) == 0:
                self.me = 0
            else:
                self.me = 1
            
    def fit(self, img, *args):
        if self.n_bands == 0:
            self.n_bands = img.shape[-1]
            
        self.mean = np.mean(img, axis =0)
        if (self.alg_type // 2) > 0:
            self.proj = alg_types[self.alg_type](img, self.n_bands, self.me)
        else:
            self.proj = alg_types[self.alg_type](img, self.n_bands)
        #self.inv = pinv(self.proj)
    
    def set_r(self, r):
        if r == -1:
            r = self.n_bands
        return r
    
    def transform(self, img, r=-1):
        r = self.set_r(r)
        if r == self.n_bands:
            return self.proj@(img-self.mean).transpose()
        else:
            proj = self.proj[:r]
            return proj@(img-self.mean).transpose()
        
    def inverse_transform(self, tr_img, r=-1):
        r = self.set_r(r)
        inv = pinv(self.proj[:r])
        return (inv@tr_img).transpose()+ self.mean
        
    def endecode(self, img, r=-1):
        r = self.set_r(r)
        return self.inverse_transform(self.transform(img, r), r)
            
    def save(self, file_name):
        np.savez(file_name, proj=self.proj,
                 mean=self.mean,
                 alg_type=[self.alg_type])
        
    def trained_weights(self):
        props = {
            'projection' : self.proj,
            'mean': self.mean
         }
        return props

    
    def load(self, file_name):
        loaded = np.load(file_name)
        self.proj = loaded['proj']
        self.n_bands = self.proj.shape[-1]
        self.mean = loaded['mean']
        self.alg_type = loaded['alg_type'][0]
        if (self.alg_type // 2) > 0:
            if (self.alg_type % 2 ) == 0:
                self.me = 0
            else:
                self.me = 1

    def load_from_props(self, projection, mean):
        #meta parameters already defined, should only be used when they are known
        self.proj = projection
        self.n_bands = self.proj.shape[-1]
        self.mean = mean
