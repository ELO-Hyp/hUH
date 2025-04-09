import collections.abc
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.optimize as opt
import sklearn.svm as svm
import sklearn.decomposition as de
import sklearn.utils.extmath as ema
import sklearn.neighbors as nei
import tables as tab
import src.DR.nmf as nmf
import src.DR.osp as osp
import time  
import copy
import collections


def flat(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flat(i)]
    else:
        return [x]

class Node():
    def __init__(self, spatial_map, classifier, status=True):
        self.map = spatial_map.astype(np.float16)
        self.classifier = classifier
        self.classifier_r = classifier
        self.classifier_n = (classifier.T/(np.sum(classifier**2)**(1/2))).T
        self.beta_expectation = [1e-3]
        self.beta_probability = [1e-3]
        self.bonus_boost = 0
        self.sparsity_balance = 0
        #elf.splitter = None
        self.status=True # True = open, False = closed
        self.mu = 0
        self.reg = 0 

def svm_to_endmembers(coef, intercept, data):
    abs_coef = np.sqrt(np.sum(coef**2))
    coef_norm = coef/abs_coef
    # adjust data to account for the intercept offset 
    shifted_data = (data.T + (intercept/abs_coef*coef_norm).T).T
    # project data onto the SVM plane
    proj_coefficients = shifted_data@coef_norm.T
    # remove the part of the data that is out-of-plane
    planar_data = data - proj_coefficients*coef_norm
    # average the planar data to find the mean
    endmember_mean = np.mean(planar_data, axis=0)
    # get the two endmembers
    endmember0 = endmember_mean - coef_norm/(2*abs_coef)
    endmember1 = endmember_mean + coef_norm/(2*abs_coef)
    #if np.isnan(endmember0).any():
    #    print(abs_coef, coef_norm, endmember_mean, proj_coefficients, shifted_data)
    return endmember0, endmember1

class DEH():
    def __init__(self, no_negative_residuals = True, subfit_params={}):
        #fdig, fparta, fpartb,
        '''
        fdig must be able to find endmembers of unlabelled data
        fpart must be able to both (a) train a set of weights for a given distribution 
        and (b) find the distribution for a given set of weights
        fpart can thus be a function on either labelled or unlabelled data
        '''
        self.nodes = {}
        self.use_bsp = False
        self.NR = not no_negative_residuals # negative residuals
        self.subfit_parameters = subfit_params
        self.remainder_id = 'none'
        self.splitting_size = 100000
        self.max_depth = 10
        self.max_nodes = -1
        self.root='average3'
        self.threshold = 0.1
        self.min_size = 1000
        self.tol = 0.001
        self.tolb = 0.05
        self.mb_l = -1 
        self.mb_in = -1
        self.svm='threshold'
        self.mixed_layers = True
        self.double_fit=True
        self.approach_rate=0.1
        self.spread = 2
        self.plot_size = (239,171)
        self.plot_aspect = 0.3
        self.zeta=0.1
        self.up = 0
        self.history = []
        self.S_MAX=2
        self.weight_power=2
        self.verbose = True
        self.Cnorm = 1
        self.max_iter = 200
        self.level_scaling = 1.5
        self.beta = 0.1 # rate of grad descent
        self.node_record = {}
        self.score_record = []
        self._use_norm = False
        self.aa = True
        self.uncon = False
        self.eps = 0.01
        self.subsamp = []
        self.n_update_pts = 0
        # proportion mixed pixels when adding a node
        self.mixed_pix = 0.1
        self.exempt_node = '2'
        self.exemption_rate = 0
        self.opts = {}
        self.reg=0
        self.use_bonus_boost = True
        self.training=""
        self.start=0
        self.end=0
        self.PAA_backcount=10
        self.PAA_i = 0
        self.mu = 0
        self.sparsifying=False
        self.delta_mu = 0
        self.reg_list = []
        self.a_speed = 0.0
        self.only_ends = False
        #self.sf = 253
        #return self
        
    def set_reg(self, reg_level):
        for node in self.reg_list:
            self.nodes[node].reg = reg_level
            
    def increment_mu(self):
        if self.delta_mu > 0:
            mu_max = self.mu
            for n in self.nodes:
                try:
                    A_max1 = np.max(self.nodes[n+'1'].map)
                    A_max0 = np.max(self.nodes[n+'0'].map)
                    A_max = np.minimum(A_max1, A_max0)
                    if A_max==1:
                        self.nodes[n].mu += self.delta_mu
                    mu_max = np.maximum(mu_max, self.nodes[n].mu)
                except KeyError:
                    pass
            self.mu = mu_max
        else:
            self.mu += self.delta_mu
            for n in self.nodes:
                if self.nodes[n].mu > self.mu:
                    self.nodes[n].mu = self.mu
        
    def set_mu(self, mu):
        for n in self.nodes:
            if self.nodes[n].mu <= 0:
                self.nodes[n].mu = mu
            elif mu < self.nodes[n].mu:
                self.nodes[n].mu = mu
            else: 
                try:
                    A_max1 = np.max(self.nodes[n+'1'].map)
                    A_max0 = np.max(self.nodes[n+'0'].map)
                    A_max = np.minimum(A_max1, A_max0)
                    if A_max==1:
                        self.nodes[n].mu = mu
                except KeyError:
                    pass
            self.mu = mu
            
    def use_norm(self, truth):
        if truth:
            for node in self.nodes:
                self.nodes[node].classifier = self.nodes[node].classifier_n
            self._use_norm = True
        else:
            self._use_norm = False
            for node in self.nodes:
                self.nodes[node].classifier = self.nodes[node].classifier_r

    def random_init(self, data, n_starting_pts, seed=0):
        self.parameter_initialization(data)
        np.random.seed = seed
        starting_pix = np.random.choice(len(data), n_starting_pts)
        self.subsamp = starting_pix
        self.subsamp_weights()
        self.construct_init_from_pix(data, starting_pix)
        
    def subsamp_weights(self):
        self.weights = self.full_weights[self.subsamp]
        
    def construct_init_from_pix(self, data, starting_pix):    
        groups = [[[i],0] for i in starting_pix]
        
        while len(groups) > 1:
            groups = regroup_1it(data,
                                 self.full_weights, groups)
            
        ng = groups[0][0]
        self.nodes[''] = Node(spatial_map=np.ones(len(starting_pix)),
                                  classifier=w_avg(data[flat(ng)],
                                                   self.full_weights[flat(ng)]))
        self.nodes[''].origin_pix = ng
        
        while self.nodes_2_include():
            self.add_group_layer(data, self.full_weights)
            
        self.initialize_splitters(data)
        
    def append_node_record(self):
        deh_nodes = list(self.nodes.keys())
        n_record_nodes = list(self.node_record.keys())
        for n in n_record_nodes:
            if n in deh_nodes:
                try:
                    self.node_record[n]['W'].append(copy.deepcopy(self.nodes[n].splitter[0]))
                    self.node_record[n]['h'].append(copy.deepcopy(self.nodes[n].splitter[1]))
                except AttributeError:
                    pass
                self.node_record[n]['s'].append(copy.deepcopy(self.nodes[n].classifier))
            else:
                return -1
        for n in deh_nodes:
            if n in n_record_nodes:
                pass
            else:
                self.node_record[n] = {}
                try:
                    self.node_record[n]['W'] = [copy.deepcopy(self.nodes[n].splitter[0])]
                    self.node_record[n]['h'] = [copy.deepcopy(self.nodes[n].splitter[1])]
                except AttributeError:
                    self.node_record[n]['W'] = []
                    self.node_record[n]['h'] = []
                self.node_record[n]['s'] = [copy.deepcopy(self.nodes[n].classifier)]
                
    def add_group_layer(self, dat, weights):
        nodes = list(self.nodes)
        d = self.get_depth()
        for n in nodes:
            if len(n)==d:
                if len(self.nodes[n].origin_pix)==2:            
                    pix1 = self.nodes[n].origin_pix[1]
                    self.nodes[n+'1'] = Node(spatial_map=np.ones(self.nodes[''].map.shape),
                                   classifier=w_avg(dat[flat(pix1)], weights[flat(pix1)]))
                    self.nodes[n+'1'].origin_pix = pix1
                    pix0 = self.nodes[n].origin_pix[0]
                else:
                    pix0 = self.nodes[n].origin_pix
                self.nodes[n+'0'] = Node(spatial_map=np.ones(self.nodes[''].map.shape),
                                   classifier=w_avg(dat[flat(pix0)], weights[flat(pix0)]))

                self.nodes[n+'0'].origin_pix = pix0
        
    def nodes_2_include(self):
        d = self.get_depth()
        to_include = []
        for n in self.nodes:
            if len(n)==d:
                if len(self.nodes[n].origin_pix)==2:
                    to_include.append(n)
        return to_include

    def initialize_splitters(self, dat):
        test1 = svm.LinearSVC(max_iter=1000000, dual=True, C=10**10)
        for n in self.nodes:
            if n+'1' in self.nodes:
                points = flat(self.nodes[n+'0'].origin_pix) + flat(self.nodes[n+'1'].origin_pix)
                X = dat[points]
                y = np.zeros(len(points))
                y[len(flat(self.nodes[n+'0'].origin_pix)):]=1
                test1.fit(X,y)
                self.nodes[n].splitter = test1.coef_[0], test1.intercept_[0] #BUGFIX test1.intercept_ is 1D array and initialize needs a not array value

    def trim(self, level):
        nodes = list(self.nodes.keys())
        for n in nodes:
            if len(n) > level:
                del self.nodes[n]        
        
    def check_splitting(self):
        if len(self.get_end_nodes())==self.max_nodes:
            return False
        
        if self.get_depth()==self.max_depth:
            return False
        
        if len(self.nodes_to_split())>0:
            return True
        else:
            return False
        
    def check_single_splitting(self):
        if len(self.get_end_nodes())==self.max_nodes:
            return False
        
        if len(self.nodes_to_split())>0:
            return True
        else:
            return False
        
    def parameter_initialization(self, image):
        # takes the image in the ordinary (e.g. N x b), rather than nmf-style 
        self.delta = image.sum(axis=1).mean()
        #try:
        #    self.weights = self.wf(image)
        #except AttributeError:
        #    self.set_weight_function()
        #    self.weights = self.wf(image)
        if self._use_norm:
            image = self.get_ldata(image)
        if self.aa:
            #im_norm = (image.T/(np.sum(image.astype(np.float32)**2, axis=-1))**(1/(2+self.eps))).T
            self.nodes[''] = Node(spatial_map = np.ones(image.shape[0], dtype=bool),
                              classifier = np.average(image,axis=0,weights=self.weights),
                              status=False)
        else:
            avg = np.average(image,axis=0,weights=self.weights)
            best_pix = np.argmin(np.sum((image-avg)**2, axis=-1))
            self.nodes[''] = Node(spatial_map = np.ones(image.shape[0], dtype=bool),
                              classifier = image[best_pix],
                              status=False)

    def set_weight_function(self):
        def wf(image):
            if self.weight_power > 0:
                return 1/np.sum(np.abs(image)**self.weight_power, axis=1)**(1/self.weight_power)
            else:
                return 1/np.sum(np.abs(image)**self.weight_power, axis=1)
        self.wf = wf
       
    def get_full_weights(self):
        if len(self.subsamp) > 0:
            self.weights = self.full_weights[self.subsamp]
        else:
            self.weights = self.full_weights
    
    def set_neighbor_weights(self, data):   
        ldata = self.get_ldata(data)
        rel_data = self.neighbors > -1
        expected_errs = np.zeros(data.shape[0])
        self.full_weights = np.zeros(data.shape[0]) + 1
        expected_errs[rel_data] = np.sum((ldata[rel_data] - ldata[self.neighbors[rel_data]])**2, axis=-1)
        self.full_weights[rel_data] = 1/(np.sqrt(expected_errs[rel_data])+np.mean(np.sqrt(expected_errs[rel_data])))
        
        self.full_weights[rel_data] /= self.full_weights[rel_data].mean()
        
        #if self._use_norm == False:
        self.full_weights[:] /= np.sqrt(np.sum(ldata**2, axis=1))
        
        self.get_full_weights()

    def set_neighbor_weights_memory_efficient(self, data, batch_size=10000):
        """
        Memory-efficient version of set_neighbor_weights that processes data in batches.
        
        Parameters:
        -----------
        data : ndarray
            The input data array
        batch_size : int, default=10000
            The size of batches to process at once
            
        Notes:
        ------
        This method reduces memory usage by processing data in smaller batches
        instead of loading the entire dataset into memory at once.
        """
        # Initialize weights array once
        n_samples = data.shape[0]
        self.full_weights = np.ones(n_samples)
        
        # Process all data to get norms for normalization later
        if self._use_norm:
            # Get ldata in batches to avoid holding entire array in memory
            data_norms = np.zeros(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = np.arange(start_idx, end_idx)
                
                # Get ldata for this batch
                batch_ldata = self.get_ldata(data[batch_indices])
                
                # Calculate norms for this batch
                data_norms[batch_indices] = np.sqrt(np.sum(batch_ldata**2, axis=1))
        else:
            # Use regular data norms
            data_norms = np.sqrt(np.sum(data**2, axis=1))
        
        # Get valid neighbor indices
        rel_data_indices = np.where(self.neighbors > -1)[0]
        
        # Process in batches
        err_mean_sum = 0
        err_count = 0
        
        # First pass to calculate mean error
        for i in range(0, len(rel_data_indices), batch_size):
            batch_rel_indices = rel_data_indices[i:i+batch_size]
            
            # Get data and neighbor data for this batch
            if self._use_norm:
                batch_data = self.get_ldata(data[batch_rel_indices])
                batch_neighbor_data = self.get_ldata(data[self.neighbors[batch_rel_indices]])
            else:
                batch_data = data[batch_rel_indices]
                batch_neighbor_data = data[self.neighbors[batch_rel_indices]]
            
            # Calculate squared differences
            diffs = batch_data - batch_neighbor_data
            squared_diffs = np.sum(diffs**2, axis=1)
            
            # Add to running mean calculation
            err_mean_sum += np.sum(np.sqrt(squared_diffs))
            err_count += len(batch_rel_indices)
        
        # Calculate the mean error
        if err_count > 0:
            err_mean = err_mean_sum / err_count
        else:
            err_mean = 1.0
            
        # Second pass to set weights
        for i in range(0, len(rel_data_indices), batch_size):
            batch_rel_indices = rel_data_indices[i:i+batch_size]
            
            # Get data and neighbor data for this batch
            if self._use_norm:
                batch_data = self.get_ldata(data[batch_rel_indices])
                batch_neighbor_data = self.get_ldata(data[self.neighbors[batch_rel_indices]])
            else:
                batch_data = data[batch_rel_indices]
                batch_neighbor_data = data[self.neighbors[batch_rel_indices]]
            
            # Calculate squared differences
            diffs = batch_data - batch_neighbor_data
            squared_diffs = np.sum(diffs**2, axis=1)
            
            # Set weights for this batch
            self.full_weights[batch_rel_indices] = 1/(np.sqrt(squared_diffs) + err_mean)
        
        # Normalize weights
        if len(rel_data_indices) > 0:
            rel_weights_mean = np.mean(self.full_weights[rel_data_indices])
            if rel_weights_mean > 0:
                self.full_weights[rel_data_indices] /= rel_weights_mean
        
        # Apply data norms normalization
        # Avoid division by zero
        nonzero_mask = data_norms > 1e-10
        self.full_weights[nonzero_mask] /= data_norms[nonzero_mask]
        
        # Update weights attribute based on subsample if needed
        self.get_full_weights()
        
        return self.full_weights
        
    def print_nmax(self, level):
        return 0
        n_nodes = 0
        l_nodes = []
        maxes = []
        for node in self.nodes:
            if len(node)==level:
                n_nodes += 1
                l_nodes.append(node)
        S = np.zeros((n_nodes,) + self.nodes[l_nodes[0]].map.shape)
        for i, node in enumerate(l_nodes):
            S[i,:] = self.nodes[node].map[:]
            maxes.append(self.nodes[node].map.max())
        sums = np.sum(S, axis = 0)
        if self.verbose:
            print(n_nodes, 'nmax ',S.max(), maxes)

    def update_lower_level_spatial(self, level, data):
        L_nodes = [i for i in self.nodes if len(i)==level]
        endnodes = self.get_end_nodes()
        for node in L_nodes:
            if len(node)==level:
                if node + '1' in self.nodes:
                    nmap = self.nodes[node].map
                    omap = (self.nodes[node+'0'].map + self.nodes[node+'1'].map)
                    og0 = omap > 0
                    
                    S = np.array([self.nodes[node+'0'].map,
                                   self.nodes[node+'1'].map])
                    S[:,og0] *= nmap[og0]/omap[og0]
                    
                    n_endnodes = 0
                    LN = len(node)
                    for enode in endnodes:
                        if enode[:LN]==node:
                            n_endnodes += 1
                            
                    for enode in endnodes:
                        if enode[:LN]==node:
                            smap = np.copy(self.nodes[enode].map)
                            smap[og0] *= nmap[og0]/omap[og0]
                            ii = int(enode[LN])
                            S = self.setS_to_threshold(S, ii, np.minimum(smap, self.nodes[enode].map))
                  
                    self.nodes[node+'0'].map = S[0]
                    self.nodes[node+'1'].map = S[1]
    
    def bipropagate(self, data, vary_spectra=True):
        depth = self.get_depth()
        for level in range(depth):
            self.spectral_from_spatial(level, data)
            self.spatial_from_spectral(level, data)
            if level < (depth -1):
                self.update_lower_level_spatial(level+1, data)
            if vary_spectra:
                self.update_spectra(data, level, utype=self.root)

    def train_clap_cycle(self, image):
        orig_S = self.end_node_map()
        self.full_fit(image)
        self.update_intermediate_node_maps()
        self.fix_endnode_maps()
        #self.trim_empty_nodes()
        self.update_intermediate_node_maps()
        self.display_level(self.get_depth())
        self.bipropagate(image)
        #self.propagate_spatial_upward(image)
        #self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        #self.train_open_nodes(image)
        new_S = self.end_node_map()
        delta = (np.abs(orig_S - new_S)).sum(axis=0)
        self.history[-1].append(delta.sum()/orig_S.shape[1])
        if self.verbose:
            print("delta: ", (delta.sum()/orig_S.shape[1]))
        self.trim_empty_nodes()
        return delta.sum()/orig_S.shape[1]
    
    def one_loop_clap(self, image):
        self.training = 'one_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_clap_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            self.train_clap_cycle(image)
            self.display_level(self.get_depth())
            
    def opt_spectra_1step(self, pixels, level):
        nodes_at_level = [i for i in self.nodes if len(i)==level]
        for i in nodes_at_level:
            self.nodes[i].update = np.zeros_like(self.nodes[i].classifier)
        
        self.predict(pixels)
        e = self.remainder_at_level(image, level)
        
    def train(self, image, loop='closed', cycle='clap'):
        self.start = time.time()
        if loop=='open':
            self.feed_forward_train(image)
        elif loop=='one_loop':
            if cycle=='slap':
                self.one_loop_train(image)
            elif cycle=='clap':
                self.one_loop_clap(image)
        elif loop =='closed':
            if cycle=='slap':
                self.train_slap(image)
            elif cycle=='clap':
                self.train_clap(image)
    
    def train_clap(self, image):
        self.training = 'closed loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.train_clap_cycle(image)
        delta = 1
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            delta = 1
            while delta > self.tolb:
                delta=self.train_clap_cycle(image)
                self.display_level(self.get_depth())
    
    def propagate_spatial_upward(self, data):
        depth = self.get_depth()
        for level in range(depth-1, -1, -1):
            #print(level)
            self.spectral_from_spatial(level, data)
            
    def propagate_spectral_downward(self, data, vary_spectra=True):
        depth = self.get_depth()
        for level in range(depth):
            if self.verbose:
                print("level ", level)
            self.spatial_from_spectral(level, data)
            if vary_spectra:
                self.update_spectra(data, level, utype=self.root)
                     
    def update_all_spectra(self, data, utype='spatial'):
        depth = self.get_depth()
        for level in range(depth+1):
            self.update_spectra(data, level, utype)
          
    def ff_norm_S(self):
        end_nodes = self.get_end_nodes()
        S = np.zeros((len(end_nodes), len(self.nodes[''].map)), dtype=np.float32)
                     
        for i, x in enumerate(self.end_nodes):
            S[i] = self.nodes[x].map
        
        S /= S.sum(axis=0)
        
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
                     
    def feed_forward_train(self, image):
        self.training = 'open_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.nodes[''].splitter = classifiers_2_svm(self.nodes['0'].classifier,
                                                            self.nodes['1'].classifier)
        if self.verbose:
            print(self.nodes_to_split())
        while (len(self.nodes_to_split()) > 0) & (self.get_depth() < self.max_depth):
            o_depth = self.get_depth()
            self.add_another_node_layer(image)
            self.ff_norm_S()
            for nodeID in self.nodes:
                if len(nodeID)==o_depth:
                    try:
                        self.nodes[nodeID].splitter = classifiers_2_svm(self.nodes[nodeID+'0'].classifier,
                                                            self.nodes[nodeID+'1'].classifier)
                    except KeyError:
                        pass
            self.update_spectra(image, self.get_depth(), utype=self.root)
            rem = self.remainder_at_level(image, self.get_depth())
            err = np.sum(self.weights*np.sum(rem**2, axis=-1))
            self.history.append([time.time()-self.start,
                                 self.get_depth(),
                                 err
                                ])
            if self.verbose:
                self.display_level(self.get_depth())
            
    def one_loop_train(self, image):
        self.training = 'one_loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_1_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth())
            self.add_another_node_layer(image)
            self.train_1_cycle(image)
            self.display_level(self.get_depth())
            #self.full_fit(image)
            
    def forward_svm_train(self, image):
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        self.train_1_cycle(image)
        self.display_level(1)
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            #self.full_fit(image)
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            self.update_intermediate_node_maps()
            self.display_level(self.get_depth())
            self.propagate_spatial_upward(image)
            self.propagate_spectral_downward(image)
            self.propagate_one_child_nodes()
            self.display_level(self.get_depth())
        
    def train_slap(self, image):
        self.training = 'closed loop'
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.train_1_cycle(image)
        delta = 1
        while (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth):
            self.update_spectra(image, self.get_depth(), utype=self.root)
            self.add_another_node_layer(image)
            delta = 1
            while delta > self.tolb:
                delta=self.train_1_cycle(image)
                self.display_level(self.get_depth())
            #self.full_fit(image)
            #print('full fit')
            #self.display_level(self.get_depth())
        #self.train_1_cycle(image)
    
    def train_cycle(self, image):
        delta = 1
        while delta > self.tolb:
            delta=self.train_1_cycle(image)
            if self.verbose():
                self.display_level(self.get_depth())
        
    def stablize_cycle(self, image):
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        S = []
        for i in self.end_nodes:
                S.append(self.nodes[i].map)
        S = np.array(S)
        S = self.set_to_threshold(S)
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
        orig_S = self.end_node_map()
        self.update_intermediate_node_maps()
        self.propagate_spatial_upward(image)
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()

    def predict(self, image):
        #if image.shape[0] != self.weights.shape[0]:
        self.weights = self.wf(image)
        for n in self.nodes:
            self.nodes[n].map = np.ones(image.shape[0])
        self.remainder_at_level(image, 0)
        self.propagate_spectral_downward(image, vary_spectra=False)
        self.propagate_one_child_nodes()
        S = []
        self.get_end_nodes()
        for i in self.end_nodes:
            S.append(self.nodes[i].map)
        S = np.array(S)
        return S 
    
    def simple_predict(self, image, all_nodes = False, weights = (), top_map = ()):
        #if image.shape[0] != self.weights.shape[0]:
        if len(weights) == 0:
            try:
                self.weights = self.wf(image)
            except AttributeError:
                self.set_weight_function()
                self.weights = self.wf(image)
        else:
            self.weights = weights
        nodes = list(self.nodes.keys())
        nodes.sort()
        self.nodes[''].lmda = np.ones(image.shape[0], dtype=np.float32)
        for n in nodes:
            if n + '1' in nodes: #self.nodes[n].map = np.ones(image.shape[0])
                lmda = classify_from_partition(image, self.nodes[n].splitter[0],
                                    self.nodes[n].splitter[1])
                self.nodes[n+'1'].lmda = lmda.astype(np.float32)
                self.nodes[n+'0'].lmda = (1-lmda).astype(np.float32)
            elif n+'0' in nodes:
                self.nodes[n+'0'].lmda = np.ones(image.shape[0], dtype=np.float32)
        if len(top_map)==0:
            self.nodes[''].map = np.ones(image.shape[0], dtype=np.float32)
        else:
            self.nodes[''].map[:] = top_map
        self.lmda_2_map()
        
        S = []
        if all_nodes:
            nodes = list(self.nodes.keys())
            nodes.sort()
        else:
            nodes = self.get_end_nodes()
        for i in nodes:
            S.append(self.nodes[i].map)
        S = np.array(S)
        return S
    
    def lmda_2_map(self):
        d = self.get_depth()
        nodes = list(self.nodes.keys())
        for i in range(1,d+1):
            for n in nodes:
                if len(n)==i:
                    m = self.nodes[n[:-1]].map * self.nodes[n].lmda
                    self.nodes[n].map = m  
                    
    def binarize_lmdas(self):
        for n in self.nodes:
            if len(n) > 0:
                self.nodes[n].lmda = np.round(self.nodes[n].lmda)
        #        if n[-1]=='0':
        #            self.nodes[n].lmda = np.round(self.nodes[n].lmda)
        #        else:
        #            m = n[:-1]+'0'
        #            self.nodes[n].lmda = 1-np.round(self.nodes[m].lmda)

    def base_predict(self, image):
        classifiers = self.get_end_classifiers()
        try:
            S = np.float32(nmf.nmf_project(image.T, classifiers.T, delta=self.delta, verbose=True,
                      tol=self.tol))
        except AttributeError:
            self.parameter_initialization(image)
            S = np.float32(nmf.nmf_project(image.T, classifiers.T, delta=self.delta, verbose=True,
                      tol=self.tol))
        self.get_end_nodes()
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
        self.update_intermediate_node_maps()
            
        return S
        
    def train_1_cycle(self, image):
        orig_S = self.end_node_map()
        self.full_fit(image)
        self.update_intermediate_node_maps()
        self.fix_endnode_maps()
        #self.trim_empty_nodes()
        self.update_intermediate_node_maps()
        self.display_level(self.get_depth())
        #self.bipropagate(image)
        self.propagate_spatial_upward(image)
        self.propagate_spectral_downward(image)
        self.propagate_one_child_nodes()
        #self.train_open_nodes(image)
        new_S = self.end_node_map()
        delta = (np.abs(orig_S - new_S)).sum(axis=0)
        if self.verbose:
            print("delta: ", (delta.sum()/orig_S.shape[1]))
        self.trim_empty_nodes()
        return delta.sum()/orig_S.shape[1]
        
    def train_single_open_node(self, image, node):
        rimage = self.get_reduced_dataset(image, node)
        # train on rimage
        
    def upstream_nodes(self, nodeID):
        #get all possible upstream nodes
        #l = len(nodeID)
        unodes = [i for i in self.nodes if nodeID[:len(i)]==i]
        unodes.remove(nodeID)
        return unodes
    
    def downstream_nodes(self, nodeID):
        #get all possible downstream nodes
        l = len(nodeID)
        dnodes = [i for i in self.nodes if i[:l]==nodeID]
        dnodes.remove(nodeID)
        return dnodes
    
    def get_end_nodes(self):
        d = self.get_depth()
        self.end_nodes = [i for i in self.nodes if len(i)==d]
        #for node in self.end_nodes:
        #    if node + '0' in self.nodes:
        #        self.nodes[node].status = False
        #self.end_nodes = [i for i in self.nodes if self.nodes[i].status]
        return self.end_nodes
    
    def end_node_map(self):
        end_nodes = self.get_end_nodes()
        S = np.zeros((len(end_nodes), self.nodes[''].map.shape[0]), dtype=np.float32)
        for i, x in enumerate(end_nodes):
            S[i] = self.nodes[x].map
        return S
    
    def get_end_classifiers(self):
        end_nodes = self.get_end_nodes()
        self.end_classifiers = np.array([self.nodes[i].classifier for i in self.end_nodes])
        return self.end_classifiers
    
    def get_rel_spec(self, image):
        rel_pix = []
        for i in self.end_nodes:
            self.nodes[i].map[rel_pix] *= 0.01
            new_pix = np.argmin(np.abs(1-self.nodes[i].map)) 
            self.nodes[i].map[rel_pix] *= 100
            rel_pix.append(new_pix)
        rel_spec = image[rel_pix]
        return rel_spec
    
    def update_classifiers(self, data): 
        for i in self.nodes:
            self.nodes[i].classifier = self.weighted_average(i, data, 3)
    
    def update_spectra(self, data, level, utype='spatial'):
        S = []
        rel_nodes = []
        if utype=='spatial':
            for i in self.nodes:
                if len(i)==level:
                    S.append(self.nodes[i].map)
                    rel_nodes.append(i)
            S = np.array(S)
            rel_spec = np.array([opt.nnls(np.multiply(np.sqrt(self.weights), S).T,
                     (np.multiply(np.sqrt(self.weights), data.T[i])))[0] for i in range(data.shape[1])]).T
            for i, x in enumerate(rel_nodes):
                self.nodes[x].classifier = rel_spec[i]
        elif 'average' in utype:
            p = int(utype[7:])
            for i in self.nodes:
                if len(i)==level:
                    self.nodes[i].map = np.minimum(self.S_MAX, self.nodes[i].map)
                    self.nodes[i].classifier = self.weighted_average(i, data, p)
    
    def full_fit(self, image):
        classifiers = self.get_end_classifiers()
        
        if self.root == 'pixel':
            rel_pix = []
            for i in self.end_nodes:
                old = np.copy(self.nodes[i].map[rel_pix])
                self.nodes[i].map[rel_pix] = 0
                new_pix = np.argmin(np.abs(1-self.nodes[i].map)) 
                self.nodes[i].map[rel_pix] = old
                rel_pix.append(new_pix)
            rel_spec = image[rel_pix]
        elif self.root == 'model':
            rel_spec = np.copy(classifiers)
        elif 'average' in self.root:
            p = int(self.root[7:])
            rel_spec = []
            self.update_all_spectra(image, utype=self.root)
            for i in self.end_nodes:
                rel_spec.append(self.weighted_average(i, image, p))
            rel_spec = np.array(rel_spec)
            if self.mixed_layers:
                d = self.get_depth()
                npa = np.array([self.nodes[''].classifier for x in self.end_nodes])
                rel_spec = np.zeros_like(npa, dtype=np.float32)
                rel_spec[:] += npa[:]
                div = 1
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if (np.sum(self.nodes[x[:j]].classifier[:]) > 0):
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]*1.5**(j)
                            #div += (j+1)
                        else:
                            #divx = div + (j+1)
                            rel_spec[i,:] += rel_spec[i,:]*1.5**(j)
                            #div += (j+1)
                divisor = (1.5**np.arange(0,d+1)).sum()
                rel_spec /= divisor#(d+1)
                #rel_spec /= divisor
        elif self.root == 'spatial':
            S = []
            for i in self.end_nodes:
                S.append(self.nodes[i].map)
            S = np.array(S)
            S = self.set_to_threshold(S)
            if self.mixed_layers:
                self.update_intermediate_node_maps()
                self.update_all_spectra(image)
                d = self.get_depth()
                rel_spec = 0*np.array([self.nodes[x].classifier for x in self.end_nodes])
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if not np.isnan(self.nodes[x[:j]].classifier[:]).any():
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]
                        else:
                            rel_spec[i,:] *= (j+1)/j
                divisor = np.arange(0,d+1).sum()
                rel_spec /= d
            else:
                rel_spec = np.array([opt.nnls(np.multiply(np.sqrt(self.weights), S).T,
                     (np.multiply(np.sqrt(self.weights), image).T)[i])[0] for i in range(image.shape[1])]).T
                for i, x in enumerate(self.end_nodes):
                    self.nodes[x].classifier = rel_spec[i]
        print(rel_spec[:,0])
        # this nnls is kept, because it allows transition from zero, which is necessary for the feedback mechanism to work
        S = np.float32(nmf.nmf_project(image.T, rel_spec.T, delta=self.delta, verbose=True,
                      tol=self.tol, zeta=self.zeta))
        #S = self.set_to_threshold(S)
        #S = self.set_to_threshold(S, do_set=False)
        if self.double_fit:
            if False:
                self.update_intermediate_node_maps()
                self.update_all_spectra(image)
                d = self.get_depth()
                rel_spec = 0*np.array([self.nodes[x].classifier for x in self.end_nodes])
                for i, x in enumerate(self.end_nodes):
                    for j in range(1,len(x)+1):
                        if not np.isnan(self.nodes[x[:j]].classifier[:]).any():
                            rel_spec[i,:] += self.nodes[x[:j]].classifier[:]
                        else:
                            rel_spec[i,:] *= (j+1)/j
                divisor = np.arange(0,d+1).sum()
                rel_spec /= d
                out = rel_spec.T
            else:
                S[S<0] = -S[S<0]
                S = self.set_to_threshold(S)
                spec = self.update_spectra_MUR(rel_spec, S, image)
                ##out = np.array([opt.nnls(np.multiply(np.sqrt(self.weights),S).T,
                ##         (np.multiply(np.sqrt(self.weights), image.T[i])))[0] for i in range(image.shape[1])])
        
        
        
        #print(out.shape)
            #spec = out.T
            #S = np.float32(nmf.nmf_project(image.T, spec.T, delta=self.delta, verbose=True, tol=self.tol))
            S = self.update_abundances_MUR(spec, S, image)
            #S = self.set_to_threshold(S)
        S[S<0] = -S[S<0]
        S /= S.sum(axis=0)
        print('smax', S.max(), S.sum(axis=0).shape, S.sum(axis=0).max(), S.sum(axis=1).shape, S.sum(axis=1).max())
        S /= S.sum(axis=0)
        am = np.argmax(S)
        n = am // S.sum(axis=0).shape[0]
        pid = am - S.sum(axis=0).shape[0]*n
        print('smax', S.max(), np.argmax(S), n , pid, S[:,  pid])
        
        self.history.append([time.time() - self.start,
                             self.get_depth(),
                             self.get_error(spec.T, S, image)])
        
        #S = self.set_to_threshold(S)
        #S /= S.sum(axis=0)
        #out = np.array([opt.nnls(S.T,
        #             (image.T)[i])[0] for i in range(image.shape[1])])
        #spec = out
        #S = np.float32(nmf.nmf_project(image.T, spec, delta=self.delta, verbose=True, 
        #                               no_negative_residuals=(not self.NR), tol=self.tol))
        for i, x in enumerate(self.end_nodes):
            self.nodes[x].map = S[i]
    
    def get_error(self, M, S, data):
        esum = 0 
        #print(M.shape, S.shape, data.shape)
        for i in range(len(data)):
            esum += self.weights[i] * np.sum((data[i] - M@S[:,i])**2)
        return esum
    
    def fix_endnode_maps(self):
        en = self.get_end_nodes()
        for node in en:
            if self.nodes[node].map.sum()==0:
                for i in range(1, self.get_depth()):
                    twin = (node[:-i] + str(1-int(node[-i])))
                    try:
                        tnz = self.nodes[twin].map > 0
                        new_value = tnz.min()
                        amin = np.argmin(self.nodes[twin].map - new_value)
                        self.nodes[node].map[amin] = new_value
                    except KeyError:
                        pass
                    
    def update_intermediate_node_maps(self):
        depth = self.get_depth()
        unupdated = list(self.nodes.keys())
        for i in self.end_nodes:
            unupdated.remove(i)
        for level in range(depth-1,-1,-1):
            for i in self.nodes:
                if len(i)==level:
                    try:
                        self.nodes[i].map = self.nodes[i+'0'].map + self.nodes[i+'1'].map
                    except KeyError:
                        self.nodes[i].map = self.nodes[i+'0'].map
                        
    def get_twin(self, node):
        candidate = node[:-1] + str(int(not bool(node[-1])))
        if candidate in self.nodes:
            return [candidate]
        else:
            return ()
    
    def get_depth(self):
        my = list(self.nodes.keys())
        my.sort(key=len)
        self.depth = len(my[-1])
        return self.depth
    
    def remainder_at_level(self, data, level):
        if self._use_norm:
            local_data = self.get_ldata(data.astype(np.float32))
            #self.wf(local_data)
        else: 
            local_data = np.copy(data.astype(np.float32))   
        for k in self.nodes:
            if ((len(k)==level)):
                local_data -= np.outer(self.nodes[k].map,
                                       self.nodes[k].classifier)
        self.remainder = local_data
        self.remainder_id = str(level)
        return self.remainder
    
    def remainder_at_node(self, data, nodeID, residual_type='proportional'):
        '''
        FYI this function seems wrong, at least in the definition of proportional 
        '''
        if self.remainder_id!=str(len(nodeID)):
            self.remainder_at_level(data, np.max([len(nodeID), 0]))
        
        if residual_type=='proportional':
            residual = self.remainder + self.nodes[nodeID].classifier
            #(self.nodes[nodeID].map * \
            #(self.remainder + self.nodes[nodeID].classifier).T).T
        elif residual_type=='full':
            residual = (self.remainder + \
                         np.outer(self.nodes[nodeID].map,self.nodes[nodeID].classifier))
            
        non_negative_map = (residual < 0)
        #True should mean non-negative
        non_negative_map = ~(non_negative_map.max(axis=-1))
        
        return residual, non_negative_map
    
    def train_node_from_spatial(self, nodeID, data):
        try:
            for i in range(2):
                if np.sum(self.nodes[nodeID + str(i)].map)==0:
                    return -1
        except KeyError:
            pass
        try:
            try:
                #self.print_nmax(len(nodeID))
                coefs, intercept, dmap = self.partition_node(nodeID, data)
                
                if np.isnan(intercept):
                    return -1
                
                e0, e1 = svm_to_endmembers(coefs, 
                                           intercept,
                                           data[dmap])
            except ValueError:
                e0, e1 = self.nodes[nodeID].outmembers[0], self.nodes[nodeID].outmembers[1] 
            #print("intercept, ", intercept)
            
            
            r = self.approach_rate
            try:
                self.nodes[nodeID].outmembers = [r*e0.flatten() + (1-r)*self.nodes[nodeID].outmembers[0],
                                   r*e1.flatten() + (1-r)*self.nodes[nodeID].outmembers[1]]
                #self.nodes[nodeID].splitter = [self.approach_rate*coefs \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[0],
                #                               self.approach_rate*intercept \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[1]]
            except AttributeError:
                self.nodes[nodeID].outmembers = [e0.flatten(), e1.flatten()]
                
            self.nodes[nodeID].splitter = classifiers_2_svm(self.nodes[nodeID].outmembers[0],
                                                            self.nodes[nodeID].outmembers[1])
        except TypeError:
            print('typeerror')
            pass
    
    def newest_train_node_from_spatial(self, nodeID, data):
        try:
            print("trainging ", nodeID)
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            Sfit = np.array([self.nodes[nchild0].map,
                             self.nodes[nchild1].map])
            local, nnr = self.remainder_at_node(data, nodeID)
            #print(local.shape)
            #coefs, intercept, _ = self.partition_node(nodeID, data)
            #e0, e1 = svm_to_endmembers(coefs, 
            #     intercept,
            #      data[self.nodes[nodeID].map >= (1-self.threshold)])
            if self.NR:
                out = np.array([opt.nnls(np.multiply(np.sqrt(self.weights),Sfit).T,
                            (np.multiply(np.sqrt(self.weights),local).T)[i])[0] for i in range(local.shape[1])])
                out = np.array(train_reconstructing_svm(local, self.nodes[nchild0].map,
                                               self.nodes[nchild1].map, b=5000, r=0.001,
                                         maxit=10, verbose=False, size=10000, m=0))
            else:
                nonzeros = self.nodes[nodeID].map>0
                nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
                child0 = np.argsort(self.nodes[nchild0].map)[-self.min_size:]
                child1 = np.argsort(self.nodes[nchild1].map)[-self.min_size:]
                joined_pts = np.concatenate([child0, child1])
                #out = np.array([opt.nnls((Sfit.T)[nnr_merge],
                #            (local[nnr_merge].T)[i])[0] for i in range(local.shape[1])])
                #coefs, intercept, dmap = self.partition_node(nodeID, data)
                #e0, e1 = svm_to_endmembers(coefs, 
                # intercept,
                # data[dmap])
                out = np.array(train_reconstructing_svm(local[nnr_merge], self.nodes[nchild0].map[nnr_merge],
                                               self.nodes[nchild1].map[nnr_merge], b0=1000000000, r=0.000000,
                                         maxit=20, verbose=False, size=-1, m=0))
                                         #initial_conditions = out.T))
                #
                #out = np.array([opt.nnls((Sfit.T)[nnr_merge],
                #            (local[nnr_merge].T)[i])[0] for i in range(local.shape[1])])
            r = self.approach_rate
            try:
                self.nodes[nodeID].outmembers = [r*out[0] + (1-r)*self.nodes[nodeID].outmembers[0],
                                   r*out[1] + (1-r)*self.nodes[nodeID].outmembers[1]]
                #self.nodes[nodeID].splitter = [self.approach_rate*coefs \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[0],
                #                               self.approach_rate*intercept \
                #                               + (1-self.approach_rate)*self.nodes[nodeID].splitter[1]]
            except AttributeError:
                self.nodes[nodeID].outmembers = [out[0], out[1]]
            self.nodes[nodeID].splitter = classifiers_2_svm(*self.nodes[nodeID].outmembers)
            print(self.nodes[nodeID].splitter[1])
        except KeyError:
            print('keyerror')
            pass
        
    def train_node_from_spectral(self, nodeID, data):
        '''
        this is not a great name, as nothing is trained in this function
        '''
        MAX_S = 1
        try:
            local, nnr = self.remainder_at_node(data, nodeID)
            coef, intercept = self.nodes[nodeID].splitter[0], self.nodes[nodeID].splitter[1]
            #self.nodes[nodeID].map[self.nodes[nodeID].map==0] = -1
            class_map = classify_from_partition(local, coef, intercept)
            #class_map = classify_from_partition((local.T/self.nodes[nodeID].map).T, coef, intercept)
            #self.nodes[nodeID].map[self.nodes[nodeID].map==-1] = 0
            scale_map = class_map * self.nodes[nodeID].map
            S = np.array([(self.nodes[nodeID].map-scale_map).flatten(), scale_map.flatten()])
            #ensure nothing vanishes
            for i in range(2):
                if False:#S[i].sum()==0:
                    print('testing deletion code ', nodeID)
                    e_map = ((local - np.outer(self.nodes[nodeID].map,
                                               self.nodes[nodeID].classifier))**2).sum(axis=-1)
                    worst_pix = np.argmax(e_map)
                    print(S.shape)
                    S[i, worst_pix] = self.nodes[nodeID].map[worst_pix]
                    S[i, worst_pix] = self.nodes[nodeID].map[worst_pix]
            
            
            self.nodes[nodeID + '0'].map = np.minimum(MAX_S,S[0].flatten())
            self.nodes[nodeID + '1'].map = np.minimum(MAX_S,S[1].flatten())
        except AttributeError:
            #print(nodeID, "AttributeError")
            self.nodes[nodeID + '0'].map = np.minimum(MAX_S,self.nodes[nodeID].map)
            pass
        
    def old_train_node_from_spectral(self, nodeID, data):
        try:
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            spectra = np.array([self.nodes[nchild0].classifier,
                                self.nodes[nchild1].classifier])
            local, nnr = self.f(data, nodeID)
            #S = np.float32(nmf.nmf_project(local.T, spectra.T,
            #                               delta=self.delta,
            #                               mask=self.nodes[nodeID].map))
            
            pos = self.nodes[nodeID].map > 0
            S = np.zeros((2, nnr.shape[0]), dtype=np.float32)
            S[:,pos] = scaled_2class_svm(local[pos], self.nodes[nodeID].map[pos], 
                                  self.nodes[nchild0].classifier,
                                  self.nodes[nchild1].classifier)
            
            
            
            
            #if not self.NR:
            if False:
                local, nnr = self.remainder_at_node(data, nodeID,
                                                    residual_type='full')
                nnr0 = (local - np.outer(S[1],self.nodes[nchild1].classifier)).min(axis=1) > 0
                nnr1 = (local - np.outer(S[0],self.nodes[nchild0].classifier)).min(axis=1) > 0
                
                S[0,~nnr0] = 0
                S[1,~nnr0] = self.nodes[nodeID].map[~nnr0]
                
                S[1,~nnr1] = 0
                S[0,~nnr1] = self.nodes[nodeID].map[~nnr1]
                
                both_neg = np.logical_and(~nnr0, ~nnr1)
                S0_greater = S[0] > S[1]
                S[0, np.logical_and(both_neg, S0_greater)] = \
                    self.nodes[nodeID].map[np.logical_and(both_neg, S0_greater)]
                S[1, np.logical_and(both_neg, S0_greater)] = 0
                S[0, np.logical_and(both_neg, ~S0_greater)] = 0
                S[1, np.logical_and(both_neg, ~S0_greater)] = \
                    self.nodes[nodeID].map[np.logical_and(both_neg, ~S0_greater)]
                
            self.nodes[nchild0].map = S[0]
            self.nodes[nchild1].map = S[1]
        except KeyError:
            nchild0 = nodeID + '0'
            pass
    
    def post_split_apply_nnr(self, nodeID, data):
        try:
            nchild0 = nodeID + '0'
            nchild1 = nodeID + '1'
            
            spectra = np.array([self.nodes[nchild0].classifier,
                                self.nodes[nchild1].classifier])
            local, nnr = self.remainder_at_node(data, nodeID)
            
            S = np.array([self.nodes[nchild0].map,self.nodes[nchild1].map])
            nnr0 = (local - np.outer(S[1],self.nodes[nchild1].classifier)).min(axis=1) > 0
            nnr1 = (local - np.outer(S[0],self.nodes[nchild0].classifier)).min(axis=1) > 0

            S[0,~nnr0] = 0
            S[1,~nnr0] = self.nodes[nodeID].map[~nnr0]

            S[1,~nnr1] = 0
            S[0,~nnr1] = self.nodes[nodeID].map[~nnr1]

            both_neg = np.logical_and(~nnr0, ~nnr1)
            S0_greater = S[0] > S[1]
            S[0, np.logical_and(both_neg, S0_greater)] = \
                self.nodes[nodeID].map[np.logical_and(both_neg, S0_greater)]
            S[1, np.logical_and(both_neg, S0_greater)] = 0
            S[0, np.logical_and(both_neg, ~S0_greater)] = 0
            S[1, np.logical_and(both_neg, ~S0_greater)] = \
                self.nodes[nodeID].map[np.logical_and(both_neg, ~S0_greater)]
                
            self.nodes[nchild0].map = S[0]
            self.nodes[nchild1].map = S[1]
        except KeyError:
            pass

    def spectral_from_spatial(self, level, data):
        self.update_spectra(data, level)
        for k in self.nodes:
            if len(k)==level:
                if self.svm=='threshold':
                    self.train_node_from_spatial(k, data)
                elif self.svm=='all':
                    self.newest_train_node_from_spatial(k, data)
                           
    def spatial_from_spectral(self, level, data):
        for k in self.nodes:
            if len(k)==level:
                self.train_node_from_spectral(k, data)                
                
    def get_open_nodes(self):
        open_nodes = []
        for node in self.nodes:
            if node + '1' in self.nodes:
                node0split = (node + '01') in self.nodes
                node1split = (node + '11') in self.nodes
                if not (node0split or node1split):
                    open_nodes.append(node)
        lens = np.array([len(i) for i in open_nodes])
        maxl = lens.max()
        self.open_nodes = [i for i in open_nodes if len(i) == maxl]
        return self.open_nodes
    
    def train_node(self, data, node):
        i = node
        local, nnr = self.remainder_at_node(data, i)
        initial_M = np.array([self.nodes[i+j].classifier \
                              for j in ['0','1']]).T
        nonzeros = self.nodes[i].map>0
        if self.NR:
            output = nmf.nmf(local.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map,
               initial_M=initial_M, weights=self.weights)
        else:
            nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
            output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge],
               initial_M=initial_M, zeta=self.zeta, up_pix=self.up, weights=self.weights[nnr_merge])
        S = np.array((nmf.nmf_project(local.T, output[0],
                           delta=self.delta, mask = self.nodes[i].map
                                     )),
                      dtype=np.float32)
        for j in range(2):
            l = i + str(j)
            self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])
        
    def train_open_nodes(self, data):
        nodes = self.get_open_nodes()
        for i in nodes:
            print('retraining ', i)
            local, nnr = self.remainder_at_node(data, i)
            initial_M = np.array([self.nodes[i+j].classifier \
                                  for j in ['0','1']]).T
            nonzeros = self.nodes[i].map>0
            
            try:
                if self.NR:
                    output = nmf.nmf(local.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map,
                       initial_M=initial_M, weights=self.weights)
                else:
                    nnr_merge = np.minimum(nnr, nonzeros, dtype=bool)
                    output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge],
                       initial_M=initial_M, zeta=self.zeta, up_pix=self.up,
                                     weights=self.weights[nnr_merge])
                S = np.array((nmf.nmf_project(local.T, output[0],
                                   delta=self.delta, mask = self.nodes[i].map
                                             )),
                              dtype=np.float32)
                for j in range(2):
                    l = i + str(j)
                    self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])
            except ValueError:
                self.delete_node(i)
                
    def trim_empty_nodes(self):
        old_nodes = list(self.nodes.keys())
        old_nodes.sort()
        for node in old_nodes[::-1]:
            if self.nodes[node].map.sum() == 0:
                try:
                    self.delete_node(node)
                except KeyError:
                    pass
                    
    def delete_node(self, node):
        print('deleting ', node)
        #assert
        try:
            del self.nodes[node[:-1]].splitter
            print("deleted splitter from ", node[:-1])
        except AttributeError:
            print("no splitter to delete on ", node[:-1])
        all_nodes = list(self.nodes.keys())
        #print(all_nodes)
        if (node[-1]=='0')&((node[:-1] + '1') in all_nodes):
            twin = (node[:-1] + '1')
            l = len(twin)
            print(twin)
            #self.nodes.pop(node)
            for i in all_nodes:
                print(i)
                if (node in i):
                    print('inner ', i, ' ', i[:len(node)])
                    if i[:len(node)]==node:
                        print('popping ', i)
                        self.nodes.pop(i)
                        print('popped ', i)
                    else:
                        print(i, ' not popped ', i[:len(node)])
                print('done with ', i)
            for i in all_nodes:
                if (twin in i):
                    print(i)
                    if i[:len(twin)]==twin:
                        new_name = list(i)
                        new_name[l-1] = '0'
                        new_name = ''.join(new_name)
                        print("naming ", i, " as ", new_name)
                        self.nodes[new_name]=self.nodes[i]
                        self.nodes.pop(i)
            
        else:
            self.nodes.pop(node) 
            all_nodes = list(self.nodes.keys())
            for i in all_nodes:
                if (node in i):
                    if i[:len(node)]==node:
                        self.nodes.pop(i)
        
        #print(self.nodes.keys())
        
    def initialize_nodes(self, data):     
        args = np.argsort((data**2).sum(axis=-1))
        L = len(args)
        #initial_M = np.array([data[args[int(np.ceil(3*L/4))]], data[args[int(np.floor(L/4))]]]).T
        initial_M = np.array([data[args[1]], data[args[-2]]]).T
        if self.NR:
            output = nmf.nmf(data.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, weights=self.weights)#, initial_M=initial_M)#, mask=self.nodes[i].map)
        else: 
            output = nmf.nmf(data.T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
               no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
               mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, zeta=self.zeta,
                             up_pix=self.min_size, initial_M = initial_M)
                            #weights=self.weights)#, mask=self.nodes[i].map[nnr])
        #S = np.float32(nmf.nmf_project(data.T, output[0], delta=self.delta))#,
                                      #no_negative_residuals=(not self.NR)))
            
        S = scaled_2class_svm(data, self.nodes[''].map, 
                                  output[0][:,0],
                                  output[0][:,1])
        self.nodes[''].splitter = classifiers_2_svm(output[0][:,0],
                                                            output[0][:,1])
        for i in range(2):
            self.nodes[str(i)] = Node(spatial_map=S[i], classifier=output[0][:,i])
        self.nodes[''].status=False
          
    def nodes_to_split(self):
        end_nodes = self.get_end_nodes()
        splitting_nodes = []
        for node in end_nodes:
            if self.nodes[node].map.astype(np.float64).sum() > self.splitting_size:
                if np.sum(self.nodes[node].map==1) > 2:
                    splitting_nodes.append(node)
        return splitting_nodes
    
    def add_simple_initial_node(self, split_var):
        self.nodes[''].splitter = [np.zeros(split_var.shape[1]), 0]
        self.nodes['0'] = Node(spatial_map=0.5*self.nodes[''].map,
                                         classifier=copy.deepcopy(self.nodes[''].classifier))
        self.nodes['1'] = Node(spatial_map=0.5*self.nodes[''].map,
                                         classifier=copy.deepcopy(self.nodes[''].classifier))  
        
    def add_single_node(self, data, split_var=(), n_update_points=0):
        if len(split_var) == 0:
            split_var = data
        
        #determine which node to add
        maybies = []
        nodesz = list(self.nodes.keys())
        for n in nodesz:
            if n+'1' in nodesz:
                pass
            else:
                if self.nodes[n].map.sum() > self.splitting_size:
                    if np.sum(self.nodes[n].map>1-np.abs(self.eps))>=1:
                        maybies.append(n)
                
                
        maybies_copy = copy.copy(maybies)
        for m in maybies_copy:
            if len(m) > 0:
                if m[:-1] in maybies_copy:
                    maybies.remove(m)
                #elif len(m) == self.max_depth:
                #    maybies.remove(m)
        
        internal_vars = {}
        
        d = self.get_depth()
        print("D is ", d)
        ### TO CHANGE
        o_recI = [[0,0]]
        if self.get_depth() > -1:
            for m in maybies:
                if len(m) == d:
                    internal_vars[m] = self.internal_pca(m, data)#/np.sqrt(self.nodes[m].map.sum())#self.excess_residual(m, data)#self.internal_pca(m, data)
                    #np.sum(self.internal_variance(m, data))#/(self.nodes[m].map**2).sum()
                else:
                    internal_vars[m] = self.internal_pca(m + '0', data)#/np.sqrt(self.nodes[m + '0'].map.sum())#self.excess_residual(m + '0', data)#self.internal_pca(m + '0', data)
                #np.sum(self.internal_variance(m+'0', data))#/(self.nodes[m+'0'].map**2).sum()

            print(internal_vars)
 
            to_grow = max(internal_vars, key=internal_vars.get)                              
            
            
            firstnew = copy.deepcopy(self)
            firstnew.reg = 0
            firstnew.set_mu(0)
            for n in self.nodes:
                del firstnew.nodes[n]
            
            #firstnew.full_weights = self.nodes[m].map 
            #firstnew.get_full_weights()
            try:
                sel = self.nodes[to_grow].lmap > 0 
            except AttributeError:
                sel = self.nodes[to_grow].map>0.5
            sel = sel & (self.full_weights > 0)
            rel_data = data[sel]
            firstnew.full_weights = self.full_weights[sel]*self.nodes[to_grow].map[sel]
            firstnew.subsamp = []
            firstnew.get_full_weights()
            firstnew.parameter_initialization(rel_data)
            firstnew.use_bonus_boost=False
            firstnew.grow_node('')
            print("splitting ", to_grow)
            
            firstnew.switch_training(rel_data, beta=0, tol=1e-1, n_update_points=n_update_points, 
                             scaling_factor=1, sampling_points=np.arange(rel_data.shape[0]),
                             alg='simple', obj_record=o_recI, A_tol=1,
                                     split_var=split_var[self.nodes[m]==1])
            #firstnew.quick_alt(rel_data, beta=0, tol=1e-10,
            #                 n_update_points=n_update_points, scaling_factor=1,
            #                 sampling_points=np.arange(rel_data.shape[0]),
            #    alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
            #                record_weights=False)
            delta = np.sum((firstnew.nodes['0'].classifier-firstnew.nodes['1'].classifier)**2)
            firstnew.reg = - delta/8
            firstnew.quick_alt(rel_data, beta=0, tol=1e-10,
                             n_update_points=0, scaling_factor=1,
                             sampling_points=np.arange(rel_data.shape[0]),
                alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                            record_weights=False)
            #firstnew.switch_training(rel_data, beta=0, tol=1e-1, n_update_points=n_update_points, 
            #                 scaling_factor=1, sampling_points=np.arange(rel_data.shape[0]),
            #                 alg='simple', obj_record=o_recI, A_tol=1,
            #                         split_var=split_var[self.nodes[m]==1])
            
            firstnew.use_bonus_boost=True
            firstnew.reg = delta/8
            firstnew.quick_alt(rel_data, beta=0, tol=1e-10,
                             n_update_points=0, scaling_factor=1,
                             sampling_points=np.arange(rel_data.shape[0]),
                alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                            record_weights=False)
            #firstnew.switch_training(rel_data, beta=0, tol=1e-1, n_update_points=0, 
            #                 scaling_factor=1, sampling_points=np.arange(rel_data.shape[0]),
            #                 alg='simple', obj_record=o_recI, A_tol=1,
            #                         split_var=split_var[self.nodes[m]==1])

            firstnew.simple_predict(data)
            firstnew.nodes[''].map = 1.0 * sel
            firstnew.lmda_2_map()
            firstnew.display_level(1)
            #firstnew.binarize_lmdas()
            #firstnew.lmda_2_map()
            #firstnew.update_from_level_S_V(rel_data, beta = 0, alg='simple', a_len_max=0.01, n_update_points=0,
            #                 attenuation = 1, levels=(-1,), occs=(), split_var=rel_data)
            self.exempt_node=to_grow
            self.exemption_rate = 0
            self.grow_node(to_grow)
            self.nodes[to_grow].splitter = [firstnew.nodes[''].splitter[0], firstnew.nodes[''].splitter[1]]
            self.nodes[to_grow+'0'].classifier[:] = firstnew.nodes['0'].classifier
            self.nodes[to_grow+'1'].classifier[:] = firstnew.nodes['1'].classifier

            
            #determine splitting parameters
            
            #pix = self.variance_minimizers(to_grow, data)
            #self.grow_node(to_grow)
            ## TO CHANGE
            #splitter = self.variance_minimizers(to_grow, data, split_var=split_var).nodes[''].splitter
            #ldata = self.get_ldata(data)
            #print(pix[0], pix[1])

            #plt.plot(ldata[pix[0]])
            #plt.plot(ldata[pix[1]])
            #plt.show()
            #self.nodes[to_grow].splitter = splitter
            
        else:
            #initialize and train remainder splitter
            self.subsamp=[]
            self.simple_predict(data)
            
            self.get_full_weights()
          
            
            
            opts = {}
            copies = {}
            self.get_full_weights()
            r1 = np.sum(self.weights*self.remainder_at_level(data, self.get_depth()).T**2)
            
            #group all points according to projection
            self.binarize_lmdas()
            self.lmda_2_map()
            max_abs = {}
            for k, m in enumerate(maybies):
                netnew = copy.deepcopy(self)
                print("prepping", m)
                o_recI=[0]
                
                
                netnew.exempt_node = m
                #netnew.quick_alt(data, beta=0, tol=1e-10,
                #                 n_update_points=n_update_points, scaling_factor=1,
                #                 sampling_points=np.arange(data.shape[0]),
                #    alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                #                record_weights=False)

                firstnew = copy.deepcopy(netnew)
                firstnew.reg = 0
                for n in self.nodes:
                    del firstnew.nodes[n]
                #firstnew.full_weights = self.nodes[m].map 
                #firstnew.get_full_weights()
                sel = self.nodes[m].map>0
                rel_data = data[sel]
                firstnew.full_weights = self.full_weights[sel]
                firstnew.subsamp = []
                firstnew.get_full_weights()
                firstnew.parameter_initialization(rel_data)
                firstnew.grow_node('')
                if m in self.opts:
                    firstnew.nodes[''].splitter = [self.opts[m]['s'][0], self.opts[m]['s'][1]]
                    firstnew.nodes['0'].classifier[:] = self.opts[m]['c0']
                    firstnew.nodes['1'].classifier[:] = self.opts[m]['c1']

                print("training", m)
                firstnew.switch_training(rel_data, beta=0, tol=1e-1, n_update_points=n_update_points, 
                                 scaling_factor=1, sampling_points=np.arange(rel_data.shape[0]),
                                 alg='simple', obj_record=o_recI, A_tol=1,
                                         split_var=split_var[self.nodes[m]==1])
                firstnew.quick_alt(rel_data, beta=0, tol=1e-10,
                                 n_update_points=n_update_points, scaling_factor=1,
                                 sampling_points=np.arange(rel_data.shape[0]),
                    alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                                record_weights=False)
                firstnew.quick_alt(rel_data, beta=0, tol=1e-10,
                                 n_update_points=0, scaling_factor=1,
                                 sampling_points=np.arange(rel_data.shape[0]),
                    alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                                record_weights=False)
                netnew = copy.deepcopy(self)
                #netnew.rescale_all_nodes(0.1, data, less_than=False)
                netnew.grow_node(m)
                netnew.nodes[m].splitter = [firstnew.nodes[''].splitter[0], firstnew.nodes[''].splitter[1]]
                netnew.nodes[m+'0'].classifier[:] = firstnew.nodes['0'].classifier
                netnew.nodes[m+'1'].classifier[:] = firstnew.nodes['1'].classifier

                del firstnew
                
                             
                try:
                    print("starting quick alt")
                    netnew.exempt_node = m
                    S = netnew.simple_predict(data)
                    netnew.exemption_rate = 0.0
                    netnew.display_level(netnew.get_depth())
                    #netnew.full_weights = self.full_weights[class_id]
                    classifiers = np.array([netnew.nodes[n].classifier for n in netnew.nodes])
                    class_id = [np.argmin(np.sum((data-c).astype(np.float64)**2, axis=1)) for c in classifiers]
                    
                    netnew.fix_end_classifiers(self.get_ldata(data), split_var)
                    #netnew.switch_training(classifiers, beta=0, tol=1e-1, n_update_points=n_update_points, 
                    #             scaling_factor=1, sampling_points=np.arange(classifiers.shape[0]),
                    #             alg='simple', obj_record=o_recI, A_tol=1, split_var=split_var[class_id],
                    #                      only_ends=True)
                    netnew.full_weights = self.full_weights

                    netnew.quick_alt(data, beta=0, tol=1e-10,
                                 n_update_points=n_update_points, scaling_factor=1,
                                 sampling_points=np.arange(data.shape[0]),
                     alg='simple', obj_record=o_recI, max_iter=self.max_iter, 
                                record_weights=True)
                    
                    scores = np.array(netnew.score_record)
                    print(scores, "scores")
                    scores[:,0] *= len(data)
                    new_var = np.sum(netnew.weights*netnew.remainder_at_level(data, netnew.get_depth()).T**2)


                    scores[:,0] = r1 - scores[:,0]

                    scores = np.prod(scores, axis=-1)
                    best = np.argmax(scores[::-1])+2 #the 2 rectifies indexing from the end
                    
                    netnew.load_from_node_record(best)
                    S = netnew.simple_predict(data)
                    netnew.display_level(netnew.get_depth())
                    Amax = np.min(np.max(S, axis=-1))
                    opts[m] = {'s': netnew.nodes[m].splitter,
                               'c0': netnew.nodes[m+'0'].classifier,
                               'c1': netnew.nodes[m+'1'].classifier}
                    netnew.get_full_weights()
                    new_var = np.sum(netnew.weights*netnew.remainder_at_level(data, netnew.get_depth()).T**2)
                    internal_vars[m] = r1 - new_var
                    netnew.clean_maps()
                    copies[m] = netnew
                    if internal_vars[m] > 0: 
                        max_abs[m] = Amax
                    else:
                        max_abs[m] = 2-Amax
                except IndexError:
                    internal_vars[m] = 0
                    max_abs[m] = 0
                    opts[m] = {'s': netnew.nodes[m].splitter,
                           'c0': netnew.nodes[m+'0'].classifier,
                           'c1': netnew.nodes[m+'1'].classifier}
            
            
            self.opts = copy.deepcopy(opts)
            #    plt.imshow((self.nodes[m].map*delta_err).reshape(self.plot_size), aspect=self.plot_aspect)
            #    plt.colorbar()
            #    plt.show()
            #    internal_vars[m] = np.sum(self.nodes[m].map*delta_err)
            
            #ldata = self.get_ldata(data)
            #expected_errs = np.sqrt(np.sum((ldata - ldata[self.neighbors])**2, axis=-1))
            #self.weights = 1/expected_errs
            
            ##remainder = self.remainder_at_level(data, self.get_depth())
            #remainder -= remainder.min()
            #remainder += 1
            ##rem_split = self.remainder_splitter( remainder, data,
            #                                   initialized=True)
            
            #partition remainder splitter between maybies
            ##rem_split.simple_predict(data)
            #print(data[:10,0])
            ##rem_split.display_level(1)
            
            #rem2 = rem_split.remainder_at_level(remainder, 1)
            #self.get_full_weights()
            #delta_err = np.sum(self.weights*remainder.T**2, axis=0) - np.sum(self.weights*rem2.T**2, axis=0)
            #plt.imshow(delta_err.reshape(self.plot_size), aspect=self.plot_aspect)
            #plt.colorbar()
            #plt.show()
            #for m in maybies:
            #    plt.imshow((self.nodes[m].map*delta_err).reshape(self.plot_size), aspect=self.plot_aspect)
            #    plt.colorbar()
            #    plt.show()
            #    internal_vars[m] = np.sum(self.nodes[m].map*delta_err)
                
            print(internal_vars)
            print(max_abs)
            metrics = {m:internal_vars[m]*max_abs[m] for m in internal_vars}
            print(metrics)
            to_grow = max(metrics, key=metrics.get)
            #abundances = np.array([max_abs[m] for m in max_abs])
            #if np.sum(abundances==1) > 1:
            #    better_vars = { m:internal_vars[m] for m in max_abs if max_abs[m]==1}
            #    to_grow = max(better_vars, key=better_vars.get)
            #else:
            #    to_grow = max(max_abs, key=max_abs.get)
            self.nodes = copies[to_grow].nodes
            #self.grow_node(to_grow)
            #self.nodes[to_grow].splitter = opts[to_grow]['s']
            #self.nodes[to_grow+'0'].classifier0 = opts[to_grow]['c0']
            #self.nodes[to_grow+'1'].classifier1 = opts[to_grow]['c1']
        
    def clean_maps(self):
        for n in self.nodes:
            self.nodes[n].map = np.array([], dtype=np.float16)
        
    def grow_node(self, to_grow):
        d = self.get_depth()
        if len(to_grow)==d:
            end_nodes = self.get_end_nodes()
            for n in end_nodes:
                self.nodes[n+'0'] = Node(spatial_map=self.nodes[n].map,
                                         classifier=copy.deepcopy(self.nodes[n].classifier_r))
            self.nodes[to_grow +'0'].classifier_n = (self.nodes[to_grow].classifier_n)
            #cr = self.nodes[to_grow +'0'].classifier
            #self.nodes[to_grow +'0'].classifier_n = cr / np.sqrt(np.sum(cr**2)) 
        self.nodes[to_grow +'1'] = Node(spatial_map=self.nodes[to_grow].map,
                                        classifier= copy.deepcopy(self.nodes[to_grow].classifier_r))
        self.nodes[to_grow +'1'].classifier_n = copy.deepcopy(self.nodes[to_grow].classifier_n)
        
        if len(to_grow)<(d-1):
            n_to_add = to_grow + '10'
            while len(n_to_add) <= d:
                self.nodes[n_to_add] = Node(spatial_map=self.nodes[to_grow].map, 
                                            classifier = copy.deepcopy(self.nodes[to_grow+'1'].classifier_r))
                self.nodes[n_to_add].classifier_n = copy.deepcopy(self.nodes[to_grow+'1'].classifier_n)
                n_to_add += '0'
        else:
            self.nodes[to_grow +'0'].classifier_n = copy.deepcopy(self.nodes[to_grow].classifier_n)
            #cr = self.nodes[to_grow +'0'].classifier
            #self.nodes[to_grow +'0'].classifier_n = cr / np.sqrt(np.sum(cr**2)) 
        
        if self._use_norm:
            for n in self.nodes:
                self.nodes[n].classifier = copy.deepcopy(self.nodes[n].classifier_n)
                
        self.nodes[to_grow].splitter = [0*self.nodes[to_grow].classifier,0]
        #self.rescale_node(0.5, to_grow, split_var, less_than=True)
        #self.simple_predict(split_var)
        #self.display_level(self.get_depth())
        #self.center_and_blur_node(node=to_grow, split_var=split_var)
        #self.simple_predict(data)
        #print(data[:10,0])
        #self.display_level(1)
    
    def rsi(self):
        rsplitter = DEH(no_negative_residuals=True)
        rsplitter.splitting_size = self.splitting_size
        rsplitter.max_depth=1
        rsplitter.max_iter=0
        rsplitter.max_nodes=2
        rsplitter.plot_size=self.plot_size
        rsplitter.plot_aspect= self.plot_aspect
        rsplitter.weight_power=0
        rsplitter.eps=0
        rsplitter.aa=True
        rsplitter.uncon = False
        rsplitter.use_norm(False)
        rsplitter.full_weights = self.full_weights
        rsplitter.subsamp = []
        rsplitter.weights = self.full_weights
        rsplitter.n_update_pts = self.n_update_pts
        return rsplitter
        
    def remainder_splitter(self, remainder, split_var, tol=(1e-2, 1e-3), A_tol=0,
                          initialized=False, nmap=(), aa=False, uncon=True):
        rsplitter = self.rsi()
        rsplitter.neighbors=quick_nn(remainder.reshape(self.plot_size+(-1,)),
                                     k_size=3).flatten()
        rsplitter.parameter_initialization(remainder)
        if len(nmap)==len(remainder):
            rsplitter.nodes[''].map[:] = nmap
        if initialized:
            rsplitter.add_single_node(remainder, split_var)
            #plt.plot(rsplitter.nodes[''].splitter[0])
            #plt.show()
            #print(split_var[:10,0])
            #rsplitter.simple_predict(split_var)
            #rsplitter.display_level(1)
        else:
            rsplitter.add_simple_initial_node(split_var)
            #a, b = np.argsort(remainder[:,1])[len(split_var)//2], np.argsort(remainder[:,0])[len(split_var)//2]
            #rsplitter.nodes['0'].classifier = remainder[a]
            #rsplitter.nodes['1'].classifier = remainder[b]
            #split = classifiers_2_svm(split_var[a],
            #                         split_var[b])
            #rsplitter.nodes[''].splitter = split
            s = rsplitter.nodes[''].splitter
            rsplitter.nodes[''].splitter = [s[0], 0.001]
            #rsplitter.simple_predict(split_var)
            #rsplitter.display_level(1)
            #plt.title("now")
            
        #rsplitter.nodes[''].splitter = [0*rsplitter.nodes[''].splitter[0],
        #                               0*rsplitter.nodes[''].splitter[1]]
        r_rec=[]
        rsplitter.uncon = uncon
        rsplitter.aa=aa
        rsplitter.simple_predict(split_var)
        rsplitter.switch_training(remainder, beta=0, tol=tol[1], 
                                 n_update_points=self.n_update_pts, 
                                 scaling_factor=1,
                                 sampling_points=np.arange(len(remainder)),
                                 alg='simple', 
                                 obj_record=r_rec,
                                 split_var=split_var)
        
        
        
        #rsplitter.nodes[''].splitter = [0.5*rsplitter.nodes[''].splitter[0],
        #                               0.5*rsplitter.nodes[''].splitter[1]]

    
        return rsplitter
    
    def set_splitter_midpoint(self, node, data):
        print("split start", self.nodes[node].splitter[1])
        splitter = self.nodes[node].splitter
        split_val = classify_from_partition(data, splitter[0], splitter[1])
        split_arg = np.argsort(split_val)
        cumulative = np.cumsum(self.nodes[node].map[split_arg].astype(np.float64))
        midpoint = np.argmin((cumulative-self.nodes[node].map.astype(np.float64).sum()/2)**2)
        plt.plot(cumulative)
        plt.show()
        print(midpoint)
        self.nodes[node].splitter = [
            self.nodes[node].splitter[0], 
            self.nodes[node].splitter[1]-0.5*split_val[split_arg[midpoint]]                         
                                    ]
        print("split end", self.nodes[node].splitter[1])
    
    def get_ldata(self, data):
        if self._use_norm:
            return (data.T/(np.sum(data**(2+self.eps), axis=1))**(1/(2))).T
        else:
            return data
    
    def add_another_node_layer_simple(self, data):
        to_split = self.nodes_to_split()
        level = self.get_depth()
        lowest_nodes = [i for i in self.nodes if len(i)==level]
        #print(lowest_nodes)
        eL = self.remainder_at_level(data, level)    
        self.simple_predict(data)
        ldata = self.get_ldata(data)
        
        for i in lowest_nodes:
            if i in to_split:
                print("splittting ", i)
                #pix = self.variance_minimizers(i, data)
                #_, pix = osp.atgp((self.nodes[i].map**8*(data.T/np.sum(data**2, axis=1))).T, n=2)
                #pos = np.ones(len(eL), dtype=bool)#np.min(self.nodes[i].classifier - eL, axis=1) > 0
                #big_err = np.argmax(np.sum(eL[pos]**2, axis=1))
                #args = np.argsort(((data - self.nodes[i].classifier)**2).sum(axis=1))
                self.nodes[i].splitter = classifiers_2_svm(ldata[pix[0]],
                                                           ldata[pix[1]])
                self.nodes[i+'0'] = Node(classifier=data[pix[0]], spatial_map=np.ones(data.shape[0]))
                self.nodes[i+'1'] = Node(classifier=data[pix[1]], spatial_map=np.ones(data.shape[0]))
            else: 
                self.nodes[i+'0'] = Node(classifier=self.nodes[i].classifier,
                                         spatial_map=np.ones(data.shape[0]))
        self.use_norm(self._use_norm)        
        self.simple_predict(data)
    
    def add_another_node_layer(self, data):
        to_split = self.nodes_to_split()
        level = self.get_depth()
        lowest_nodes = [i for i in self.nodes if len(i)==level]
        #print(lowest_nodes)
        for i in lowest_nodes:
            if i in to_split:
                print("splittting ", i)
                local, nnr = self.remainder_at_node(data, i)
                local = (self.nodes[i].map*local.T).T
                nonzeros = self.nodes[i].map>0
                if self.NR:
                    output = nmf.nmf(local[nonzeros].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                           no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=0,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nonzeros],
                                    weights=self.weights)
                    S = np.array((nmf.nmf_project(local.T, output[0],
                             delta=self.delta, mask = self.nodes[i].map)), dtype=np.float32)
                else: 
                    nnr_merge = (np.minimum(nnr, nonzeros, dtype=bool)).flatten()
                    #print(nnr_merge.sum(), nnr_merge)
                    #l_vals = np.multiply(local[nonzeros].T, local[nonzeros].T).sum(axis=1)
                    #selected = np.argsort(self.nodes[i].map[nonzeros])[-2:]
                    avg = self.nodes[i].classifier
                    l_vals = np.sum((local[nonzeros]-avg)**2, axis=-1)
                    selected = np.argsort(l_vals)[:2]#self.nodes[i].map[nonzeros])[-2:]
                    initial_M = local[nonzeros][selected].T#/self.nodes[i].map[nonzeros][selected]
                    if np.isnan(initial_M).any():
                        initial_M = data[nonzeros][selected].T

                    output = nmf.nmf(local[nnr_merge].T, 2, verbose=True, maxiter=self.max_iter, delta=self.delta, 
                       no_S_smoothness=True, alpha=0.00, no_M_smoothness=True, tol=self.tol, version=4,
                       mb_in=self.mb_in, mb_l=self.mb_l, beta=0.00, mask=self.nodes[i].map[nnr_merge], zeta=self.zeta,
                                    up_pix=self.min_size,  weights=self.weights[nnr_merge])#, initial_M = initial_M)
                    
                    pos = self.nodes[i].map > 0
                    S = np.zeros((2, nnr.shape[0]), dtype=np.float32)
                    S[:,pos] = scaled_2class_svm(local[pos], self.nodes[i].map[pos], 
                                  output[0][:,0],
                                  output[0][:,1])
                    self.nodes[i].splitter = classifiers_2_svm(output[0][:,0],
                                                            output[0][:,1])
                    #S = svm_from_classifiers(local.T, output[0][:,0], output[0][:,1])
                        #np.array((nmf.nmf_project(local.T, output[0],#output[0],
                        #         delta=self.delta, mask = self.nodes[i].map,
                        #         no_negative_residuals=False)), dtype=np.float32)
                for j in range(2):
                    l = i + str(j)
                    #Sprint(l)
                    self.nodes[l] = Node(spatial_map=S[j], classifier=output[0][:,j])#initial_M[:,j])
                self.nodes[i].outmembers= output[0].T
            else:
                l = i + '0'
                self.nodes[l] = Node(spatial_map=self.nodes[i].map, classifier=self.nodes[i].classifier)
            self.nodes[i].status=False
                
    def get_one_child_nodes(self):
        one_child_nodes =  []
        for node in self.nodes:
            if (node + '0' in self.nodes)&(node + '1' not in self.nodes):
                one_child_nodes.append(node)
        one_child_nodes.sort(key=len)
        return one_child_nodes
    
    def propagate_one_child_nodes(self, level=False):
        ocns = self.get_one_child_nodes()
        if level:
            ocns_true = [i for i in ocns if len(i)==level]
            ocns = ocns_true
        for node in ocns:
            self.nodes[node+'0'].map = self.nodes[node].map
            self.nodes[node+'0'].classifier = self.nodes[node].classifier
            
    def weighted_average(self, node, data, p):
        spatial = self.nodes[node].map
        weight_sum = (np.multiply(self.weights, (1-np.abs(1-spatial))**p)).sum()
        weight_spectra = (np.multiply(self.weights, (1-np.abs(1-spatial))**p)*data.T).sum(axis=-1)
        if weight_sum > 0:
            return (weight_spectra / weight_sum)
        else:
            return 0*weight_spectra

    def setS_to_threshold(self, S_to_set, sseti, S_to_read):
        n_max = self.min_size
        
        maxes = np.argsort(S_to_read)[-(n_max):]
        below_threshold = np.zeros_like(S_to_read, dtype=bool)# < (1-self.threshold)) & (S[i] > 0)
        below_threshold[maxes] = ((S_to_read[maxes]) < (1-self.threshold)) & (S_to_read[maxes] > 0)
        #print(maxes.shape, below_threshold.shape)
        delta = (1-self.threshold)/S_to_read[below_threshold] 
        for j in range(len(S_to_set)):
            if j != sseti:
                S_to_set[j, below_threshold] *= (self.threshold)/np.maximum(1-S_to_set[sseti, below_threshold],1e-16)
        S_to_set[sseti, below_threshold] *= delta
        #print(S[i, maxes])

        return S_to_set
    
    def set_to_threshold(self, S, do_set=True):
        n_max = self.min_size
        if not do_set:
            for i in range(len(S)):
                maxes = np.argsort(S[i])[-(n_max):]
                #print(S[i, maxes])
                #print(S.shape)
                
        for i in range(len(S)):
            maxes = np.argsort(S[i])[-(n_max):]
            #print(S[i, maxes])
            below_threshold = np.zeros_like(S[i], dtype=bool)# < (1-self.threshold)) & (S[i] > 0)
            below_threshold[maxes] = ((S[i,maxes]) < (1-self.threshold)) & (S[i, maxes] > 0)
            #print(maxes.shape, below_threshold.shape)
            delta = (1-self.threshold)/S[i, below_threshold] 
            for j in range(len(S)):
                if j != i:
                    S[j, below_threshold] *= (self.threshold)/(1-S[i, below_threshold])
            S[i, below_threshold] *= delta
            #print(S[i, maxes])
            
        return S

    def display_level(self, level, original=''):
        count=0
        for i in self.nodes:
            if len(i)==level:
                if i[:len(original)]==original:
                    count += 1
        if self.verbose:
            print(count)

        fig, ax = plt.subplots(count, figsize=(8,2*count))
        for i, a in enumerate(ax):
            a.set_xticks([])
            a.set_yticks([])

        counter = 0
        for i in self.nodes:
            if len(i)==level:
                if i[:len(original)]==original:
                    ax[counter].imshow(np.rot90(self.nodes[i].map.reshape(self.plot_size)),
                                        aspect=self.plot_aspect, vmin=0, vmax=1, interpolation='nearest')
                    ax[counter].set_ylabel(self.nodes[i].map.astype(np.float32).sum())
                    ax[counter].set_title(i + " {:.2f}".format(self.nodes[i].map.max()))
                    counter += 1
        plt.show()

    def partition_node(self, node, data):
        min_size = self.min_size#int(np.ceil(np.max([self.min_size / 2**(len(node)), 1])))
        print("partitioning ", node)
        if (node +'0' and node+'1') in self.nodes:
            lima = self.threshold 
            limb = np.sort(np.abs(1-self.nodes[node +'0'].map))[(min_size+1)]
            if limb == 0:
                try:
                    limb = self.nodes[node+'0'].map[self.nodes[node+'0'].map>0].min()
                except ValueError:
                    other_min = self.nodes[node+'1'].map[self.nodes[node+'1'].map>0].min()
                    arg = np.argmin(np.abs(other_min-self.nodes[node+'1'].map))
                    self.nodes[node+'0'].map[arg] = self.nodes[node+'1'].map[arg]
                    self.nodes[node+'1'].map[arg] = 0
            print(lima, limb)
            #print(np.sort(self.nodes[node +'0'].map)[-(min_size+1):])
            map0 = np.array(np.abs(1-self.nodes[node +'0'].map) <= min(lima, limb))
            #if map0.sum()==0:
            #    arg = np.argmax(self.nodes[node +'0'].map)
            #    map0[arg] = True
            lima = self.threshold 
            limb = np.sort(np.abs(1-self.nodes[node +'1'].map))[(min_size+1)]
            if limb == 0:
                try:
                    limb = self.nodes[node+'1'].map[self.nodes[node+'1'].map>0].min()
                except ValueError:
                    other_min = self.nodes[node+'0'].map[self.nodes[node+'0'].map>0].min()
                    arg = np.argmin(np.abs(other_min-self.nodes[node+'0'].map))
                    self.nodes[node+'1'].map[arg] = self.nodes[node+'0'].map[arg]
                    self.nodes[node+'0'].map[arg] = 0
            print(lima, limb)
            #print(np.sort(self.nodes[node +'1'].map)[-(min_size+1):])
            map1 = np.array(np.abs(1-self.nodes[node +'1'].map) <=  min(lima, limb))
                
            Xs = data[np.logical_xor(map1, map0)]
            ys = np.zeros(np.logical_xor(map1, map0).sum(), dtype=bool)

            ys[map1[np.logical_xor(map1, map0)]]=True
            if len(set(ys.tolist())) == 1:
                return np.nan, np.nan, np.nan
            
            sdev = np.std(Xs)
            
            print("about to train the svm")
            test1 = svm.LinearSVC(max_iter=100000, dual=False, tol=1e-4, C=self.Cnorm/sdev**2,
                                  class_weight='balanced')
            test1.fit(Xs, ys)
            x_guess = test1.predict(Xs)
            print(np.mean(x_guess[ys==1]==ys[ys==1]))
            print(np.mean(x_guess[ys==0]==ys[ys==0]))
            d_out = data@test1.coef_.T + test1.intercept_
            d_out2 = classify_from_partition(data, test1.coef_,test1.intercept_)

            print("svm trained")
            return test1.coef_*(1-self.threshold), test1.intercept_*(1-self.threshold), map1 | map0
        else:
            print('node {} as no children'.format(node))
            
    def save(self, filename, title='Unmixing Hierarchy', save_labels=()):
        d = self.get_depth()
        
        class Save_Node(tab.IsDescription):
            name = tab.StringCol(itemsize=d)
            classifier = tab.Float32Col(self.nodes[''].classifier.shape)
            splitter_w = tab.Float32Col(self.nodes[''].classifier.shape)
            intercept = tab.Float32Col()
            
        if filename[-2:] != 'h5':
            print("can only save as h5 file")
            return -1
        
        h5file = tab.open_file(filename, mode='w', title=title) 
        try:
            table = h5file.create_table("/", "nodes", Save_Node)
            h5file.create_array("/", "history", self.history)
            node_save = table.row
            for node in self.nodes:
                node_save['name'] = node
                node_save['classifier'] = self.nodes[node].classifier.astype(np.float32)
                try:
                    node_save['splitter_w'] = self.nodes[node].splitter[0].astype(np.float32)
                    node_save['intercept'] = float(self.nodes[node].splitter[1])
                except AttributeError:
                    pass
                node_save.append()
                table.flush()
            table.attrs.NR = self.NR 
            table.attrs.subfit_parameters = self.subfit_parameters
            table.attrs.remainder_id = self.remainder_id
            table.attrs.splitting_size = self.splitting_size
            table.attrs.max_depth = self.max_depth
            table.attrs.root = self.root
            table.attrs.threshold = self.threshold
            table.attrs.training = self.training
            table.attrs.min_size = self.min_size
            table.attrs.tol = self.tol 
            table.attrs.tolb = self.tolb
            table.attrs.mb_l = self.mb_l  
            table.attrs.mb_in = self.mb_in
            table.attrs.svm = self.svm
            table.attrs.mixed_layers = self.mixed_layers
            table.attrs.double_fit=self.double_fit
            table.attrs.approach_rate=self.approach_rate
            table.attrs.spread = self.spread
            table.attrs.plot_x = self.plot_size[0]
            table.attrs.plot_y = self.plot_size[1]
            table.attrs.plot_aspect = self.plot_aspect
            table.attrs.weight_power = self.weight_power
            if len(save_labels)>0:
                table.attrs.cluster_labels = save_labels
            #self.plot_size = (239,171)
            #self.plot_aspect = 0.3
            table.attrs.zeta = self.zeta
            h5file.close()
        except ValueError:
            print("ValueError occured on saving")
            h5file.close()
            
        return 0
    
    def load(self, filename):
        self.h5file = tab.open_file(filename, mode='r')
        table = self.h5file.root.nodes
        for row in table.iterrows():
            node = row['name'].decode('ascii')
            self.nodes[node] = Node(np.array([0]), row['classifier'])
            if np.sum(np.abs(row['splitter_w'])) > 0:
                self.nodes[node].splitter = (row['splitter_w'], row['intercept']) 
        try:
            self.plot_size = (self.h5file.root.nodes.attrs.plot_x,
                          self.h5file.root.nodes.attrs.plot_y)
            self.plot_aspect = self.h5file.root.nodes.attrs.plot_aspect
        except AttributeError:
            pass
        self.h5file.close()
                
    def display_spectra(self, spectra_list=(), names = (), wl=(), normalizer = 1, **kwargs):
        # if self.verbose: 
        if True: #still want to see the spectra when calling the function
            if len(spectra_list)==0:
                spectra_list = [i for i in self.nodes if len(i)==self.get_depth()]
            if len(wl)==0:
                wl = np.arange(len(self.nodes[''].classifier))
            if len(names)==0:
                names = spectra_list
            for i, x in enumerate(spectra_list):
                if len(names)==len(spectra_list):
                    plt.plot(wl, self.nodes[x].classifier/normalizer, label = names[i])
                else:
                    plt.plot(wl, self.nodes[x].classifier/normalizer, label = i)
            plt.legend(**kwargs)

    def display_map(self, spectrum, ax=plt):
        ax.imshow(np.rot90(self.nodes[spectrum].map.reshape(self.plot_size)),
                  aspect=self.plot_aspect)

    def display_maps(self, spectra, names = (), figsize=()):
        count = len(spectra)
        if len(figsize) > 0:
            fig, ax = plt.subplots(count, figsize=figsize)
        else:
            fig, ax = plt.subplots(count)
        for i, spec in enumerate(spectra):
            if len(names)==len(spectra):
                name = names[i]
            else:
                name = spec
            self.display_map(spec, ax=ax[i])
            ax[i].set_title(name)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            
    def update_abundances_MUR(self, M, S, data):
        # will be performed pixel-wise
        S_out = np.zeros_like(S)
        Ma = np.append(M.T, self.delta*np.ones((1,M.shape[0])), axis=0)
        for i in range(len(S.T)):
            Da = np.append(data.T[:,i], self.delta)

            num = Da.T
            denom = Ma @ S[:,i]
            
            #print(num.shape, denom.shape)
            
            num = Ma.T @ num 
            denom = Ma.T @ denom

            S_out[:,i] = S[:,i] * num / denom
        return S_out
    
    def update_spectra_MUR(self, M, S, data):
        M_out = np.zeros_like(M)
        for i in range(len(M.T)):
            num = data[:,i]
            denom = M[:,i] @ S

            #print(num.shape, denom.shape)

            num = np.multiply(self.weights, num)
            denom = np.multiply(self.weights, denom)

            num = num @ S.T
            denom = denom @ S.T

            denom = np.maximum(denom, 1e-32)
            M_out[:,i] = M[:,i] * num / denom
        return M_out
    
    def set_lmda(self):
        for n in self.nodes:
            if len(n) > 0:
                self.nodes[n].lmda = self.nodes[n].map/np.maximum(self.nodes[n[:-1]].map, 1e-10)
            else:
                self.nodes[n].lmda = self.nodes[n].map
                
    def one_step(self, data, beta=0.1, up_level=0, scaling_factor=2, alg='complex', max_update_r=0.01):
        self.populate_fns( data, alg=alg)
        for n in self.nodes:
            self.nodes[n].h_update = 0 
            self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                              dtype=np.float32) 
        depth = self.get_depth()
        
        for i in range(depth):
            self.update_from_level(i+1, data, scaling_factor=scaling_factor, alg=alg)

        long_nodes = [n for n in self.nodes]# if len(n)>=up_level]
        self.update_ll(data, alg=alg)
        #self.lowest_spatial(data, beta=beta, alg=alg)
        
        L = len(self.nodes[''].classifier)
        for n in long_nodes:
            try:
                update = beta*self.nodes[n].W_update
                r_update = np.sum(np.abs(update))/np.sum(np.abs(self.nodes[n].splitter[0]))
                print(r_update, "r-uptdate")
                if r_update > max_update_r:
                    update *= max_update_r / r_update
                    self.nodes[n].h_update *= max_update_r / r_update
                self.nodes[n].splitter = [self.nodes[n].splitter[0], self.nodes[n].splitter[1]]
                self.nodes[n].splitter[0] += update#update
                self.nodes[n].splitter[1] += beta*self.nodes[n].h_update #* (max_update_r / r_update)
            except AttributeError:
                pass

        for n in self.nodes:
            self.nodes[n].h_update = 0 
            self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                                  dtype=np.float32) 
            try:
                del self.nodes[n].fns
            except AttributeError:
                pass
            
    def populate_fns(self, data, target='W', alg='complex'):
        if alg=='complex':
            self.predict(data)
            self.set_lmda()
        elif alg=='simple':
            self.simple_predict(data)
        
        depth = self.get_depth()
        for i in range(depth):
            #("depth of ", i)
            if target == 'W':
                self.fn_level(i, alg=alg)
            elif target == 'S':
                for n in self.nodes:
                    try: 
                        self.nodes[n].fns[n]= 0*self.nodes[''].map
                    except AttributeError:
                        self.nodes[n].fns = {n:0*self.nodes[''].map}
                self.fn_level_S( i, alg=alg)
            else:
                print("invalid target")
                
    def fn_initialize(self, level):
        nodes = [j for j in self.nodes if len(j)==(level)]
        nal = [j for j in self.nodes if len(j)==(level+1)]
        for n in nodes:
            if n+'1' in self.nodes:
                self.nodes[n].fns = {m:0*self.nodes[''].map for m in nal}
                m = n+'1'
                dl = -(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                fn = self.nodes[n].map.astype(np.float32)
                fn = np.multiply(fn, dl)
                self.nodes[n].fns[m] = fn
                m = n+'0'
                dl = (np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                fn = self.nodes[n].map.astype(np.float32)
                fn = np.multiply(fn, dl)
                self.nodes[n].fns[m] = fn
            elif n+'0' in self.nodes:
                self.nodes[n].fns = {m:0*self.nodes[''].map for m in nal}
                
    def fn_level(self, level, balanced=False, alg='complex'):
        nal = [j for j in self.nodes if len(j)==(level+1)]
        for n in self.nodes:
            if len(n)==level:
                if n+'1' in self.nodes:
                    self.nodes[n].fns = {m:0*self.nodes[''].map for m in nal}
                    m = n+'1'
                    dl = -(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                    fn = self.nodes[n].map.astype(np.float32)
                    fn = np.multiply(fn, dl)
                    if balanced:
                        fn /= np.sum(self.nodes[m].map)/np.sum(self.nodes[n].map)
                    self.nodes[n].fns[m] = fn
                    m = n+'0'
                    dl = (np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                    fn = self.nodes[n].map.astype(np.float32)
                    fn = np.multiply(fn, dl)
                    if balanced:
                        fn /= np.sum(self.nodes[m].map)/np.sum(self.nodes[n].map)
                    self.nodes[n].fns[m] = fn
            if len(n) < level:
                if alg=='complex':#nall = [j for j in self.nodes if len(j)==level-1]
                    outshape = self.nodes[n].classifier.shape + self.nodes[n].map.shape
                    try:
                        fnsum = np.array([np.outer(self.nodes[o].classifier, self.nodes[n].fns[o]) \
                                        for o in self.nodes[n].fns if len(o)==level-1]).reshape((-1,) + outshape)
                        fnsum = np.sum(fnsum, axis=0)
                    except (AttributeError, KeyError):
                        #print("it is ", n)
                        fnsum = np.zeros((len(self.nodes[n].classifier), len(self.nodes[n].map)))
                    for m in nal:
                        if m[:-1] + '1' in nal:
                            #print(n, m)
                            try:
                                fna = self.nodes[n].fns[m[:-1]]*self.nodes[m].lmda
                            except AttributeError:
                                fna = np.zeros((len(self.nodes[n].classifier), len(self.nodes[n].map)))
                            fnb = self.nodes[m[:-1]].map 
                            dl = ((-1)**int(m[-1]))*(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                            fnb = np.multiply(dl, fnb)
                            fnc = fnb*np.dot(self.nodes[m[:-1]].splitter[0], fnsum)
                            fn = fna- fnc #was +#
                            try:
                                self.nodes[n].fns[m] = fn
                            except AttributeError:
                                pass
                elif alg=='simple':
                    for m in nal:
                        if m[:len(n)]==n:
                            try:
                                fna = self.nodes[n].fns[m[:-1]]*self.nodes[m].lmda
                            except AttributeError:
                                fna = np.zeros((len(self.nodes[n].classifier), len(self.nodes[n].map)))
                            try:
                                self.nodes[n].fns[m] = fna
                            except AttributeError:
                                pass
                                
                            
        return 0
    
    def fn_level_S(self, level, alg='complex'):
        nal = [j for j in self.nodes if len(j)==(level+1)]
        for n in self.nodes:
            if len(n)==level:
                self.nodes[n].fns = {}
                for m in nal:
                    if m[:-1]==n:
                        dl = ((-1)**int(m[-1]))*(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                        fn = self.nodes[n].map.astype(np.float32)
                        fn = np.multiply(np.multiply(fn, dl), 1-fn)
                        self.nodes[n].fns[m] = fn
                    else:
                        dl = ((-1)**int(m[-1]))*(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                        fn = self.nodes[n].map.astype(np.float32)
                        fn = np.multiply(np.multiply(fn, dl),
                                         -self.nodes[m[:-1]].map.astype(np.float32))
                        self.nodes[n].fns[m] = fn
            if len(n) < level:
                if alg=='complex':
                    fnsum = np.array([np.outer(self.nodes[o].classifier, self.nodes[n].fns[o]) \
                                    for o in self.nodes[n].fns if len(o)==level-1]).reshape((-1,
                                                                  len(self.nodes[n].classifier),
                                                                   len(self.nodes[n].map)))
                    fnsum = np.sum(fnsum, axis=0)
                    for m in nal:
                        if m[:-1] + '1' in nal:
                            fna = self.nodes[n].fns[m[:-1]]*self.nodes[m].lmda 
                            fnb = self.nodes[m[:-1]].map 
                            dl = ((-1)**int(m[-1]))*(np.abs(self.nodes[m].lmda-0.5)<0.5).astype(np.float32)
                            fnb = np.multiply(dl, fnb)
                            fnc = fnb*np.dot(self.nodes[m[:-1]].splitter[0], fnsum)
                            fn = fna - fnc # was + 
                            self.nodes[n].fns[m] = fn
                elif alg=='simple':
                    for m in nal:
                        if m[:len(n)]==n:
                            fna = self.nodes[n].fns[m[:-1]]*self.nodes[m].lmda 
                            self.nodes[n].fns[m] = fna

        return 0
    
    def update_from_level(self, level, data, scaling_factor = 2, alg='complex'):
        scale = scaling_factor**level/np.sum(scaling_factor**np.arange(self.get_depth()+1))
        incl_nodes = [n for n in self.nodes if len(n) < level]
        eL = self.remainder_at_level(data, level)
        wp = np.mean(np.sum(data, axis=1)**-(self.weight_power))
        for n in incl_nodes:
            try:
                keys = self.nodes[n].fns.keys()
                incl_keys = [k for k in keys if len(k)==level]
                spectra = np.array([np.outer(self.nodes[i].classifier,
                                             self.nodes[n].fns[i]) for i in incl_keys]).sum(axis=0)
                eLn = self.remainder_at_level(data, len(n))
                
                #print('at 1')
                #print('weights', np.mean(np.abs(self.weights)))
                #print('error',np.mean(np.abs(eL)))
                #print('spectra',np.mean(np.abs(spectra)))
                pref = np.multiply(self.weights, np.multiply(eL.T, spectra).sum(axis=0))
                if alg=='complex':
                    x_in = self.nodes[n].classifier + eLn
                else:
                    x_in = data
                W = self.nodes[n].splitter[0]
                #print('pref', np.mean(np.abs(pref)))
                #print('W', W)
                #print('wp',wp)
                self.nodes[n].h_update -= scale*np.mean(pref)*np.sum(W**(2))*wp #/len(W)#W.std()/len(W)#
                self.nodes[n].W_update -= scale*np.mean(np.multiply(x_in.T, pref), axis=1)*np.sum(W**(2))*wp #*W.std()#
            except AttributeError:
                pass
            
    def update_ll(self, data, scaling_factor = 2, alg='complex'):
        level = self.get_depth()
        scale = scaling_factor**level/np.sum(scaling_factor**np.arange(self.get_depth()+1))
        incl_nodes = [n for n in self.nodes if len(n) == level-1]
        wp = np.mean(np.sum(data, axis=1)**-(2+self.weight_power))
        #if alg=='complex':
        #    self.predict(data)
        #elif alg=='simple':
        #    self.simple_predict(data)
        eL = self.remainder_at_level(data, level)
        for n in incl_nodes:
            try:
                keys = self.nodes[n].fns.keys()
                incl_keys = [k for k in keys if len(k)==level]
                spectra = np.array([np.outer(self.nodes[i].classifier,
                                         self.nodes[n].fns[i]) for i in incl_keys]).sum(axis=0)
                #print(self.weights.shape, spectra.shape, n)
                #pref = np.multiply(self.weights, np.multiply(eL.T, spectra).sum(axis=0))
                eLn = self.remainder_at_level(data, len(n))
                pref = np.multiply(self.weights, np.multiply(eL.T, spectra).sum(axis=0))
                #x_in = self.nodes[n].classifier + eLn
                if alg=='complex':
                    x_in = self.nodes[n].classifier + eLn
                elif alg=='simple':
                    x_in = data
                #wp = np.mean(self.weights**((4-self.weight_power)/self.weight_power))
                W = self.nodes[n].splitter[0]
                self.nodes[n].h_update -= scale*np.mean(pref)*np.sum(W**(2))*wp#/len(W)#*W.std()/len(W)#F*np.sum(W**2)
                self.nodes[n].W_update -= scale*np.mean(np.multiply(x_in.T, pref), axis=1)*np.sum(W**(2))*wp#*W.std()#np.sum(W**2)
            except AttributeError:
                pass
            
    def lowest_spatial(self, data, beta=0.1, alg='complex'):
        depth = self.get_depth()
        if alg=='complex':
            self.predict(data)
            self.set_lmda()
        elif alg=='simple':
            self.simple_predict(data)
        self.fn_initialize(depth-1)
        lnodes = [n for n in self.nodes if len(n)==depth-1]
        for n in lnodes:
            self.nodes[n].h_update = 0 
            self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                              dtype=np.float32) 
        self.update_ll(data)
        
        for n in lnodes:
            try:
                self.nodes[n].splitter = [self.nodes[n].splitter[0], self.nodes[n].splitter[1]]
                self.nodes[n].splitter[0] += beta*self.nodes[n].W_update#/pf
                self.nodes[n].splitter[1] += beta*self.nodes[n].h_update#/pf
                self.nodes[n].h_update = 0 
                self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                                  dtype=np.float32) 
            except AttributeError:
                pass
            
    def one_step_S(self, data, beta=0.1, max_level=-1, up_level=0, scaling_factor=2, alg='complex', max_step_r = 0.01,
                  n_update_points=0, attenuation=1, occs=(), split_var=(), levels=()):
        if self.verbose:
            print("one step S")
        if len(levels)==0:
            levels = [i for i in range(1,self.get_depth()+1)]

        self.update_from_level_S_V(data, beta=beta, alg=alg, n_update_points=n_update_points, attenuation=attenuation,
                                  levels=levels, occs=occs, split_var=split_var)

        #if alg=='complex':
        #    self.populate_fns( data, target='S', alg=alg)
        #for n in self.nodes:
        #    self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
        #                                      dtype=np.float32)

        #if max_level == -1:
        #    max_level = self.get_depth()
        #if alg=='complex':
        #    for i in range(max_level-2):
        #        self.update_from_level_S_I(i+2, data, scaling_factor=scaling_factor)
        
        
        
        #if n_update_points > 0:
        #    update_pix = np.random.choice(len(data), n_update_points)
        #    for i in range(max_level-1):
        #        self.update_from_level_S_II(i+1, data[update_pix], scaling_factor=scaling_factor)
        #else:
        #    for i in range(max_level-1):
        #        self.update_from_level_S_II(i+1, data, scaling_factor=scaling_factor)
         
        #scaling_factor = np.mean(1/self.weights)
        #long_nodes = [n for n in self.nodes if len(n) >= up_level]
        #for n in long_nodes:
        #    try:
        #        step = (beta)*scaling_factor*self.nodes[n].S_update
        #        step_mag = np.sum(np.abs(step))
                #print(step_mag)
        #        if step_mag / np.sum(self.nodes[n].classifier) > max_step_r:
        #            step = step / step_mag * max_step_r
        #        self.nodes[n].classifier += step
        #    except AttributeError:
        #        pass

        #for n in self.nodes:
        #    try:
        #        self.nodes[n].classifier[self.nodes[n].classifier<0]=0
        #        del self.nodes[n].fns
        #    except AttributeError:
        #        pass
            
    def lowest_level_S(self, data, beta=0.1, alg='complex', n_update_points=0, occs=()):
        d = self.get_depth()
        for n in self.nodes:
            self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
                                                  dtype=np.float32)
        self.update_from_level_S_V(data, beta=beta, alg=alg, n_update_points=n_update_points, occs=occs)
        #scaling_factor = np.mean(1/self.weights)
        #for n in self.nodes:
        #    try:
        #        self.nodes[n].classifier += beta*scaling_factor*self.nodes[n].S_update
        #        self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
        #                                          dtype=np.float32)
        #        self.nodes[n].classifier[self.nodes[n].classifier<0]=0
        #    except AttributeError:
        #        pass
            
    def update_from_level_S_I(self, level, data, scaling_factor = 2):
        scale = scaling_factor**level/np.sum(scaling_factor**np.arange(self.get_depth()+1))
        incl_nodes = [n for n in self.nodes if len(n) < level]
        eL = self.remainder_at_level(data, level)
        for n in incl_nodes:
            keys = self.nodes[n].fns.keys()
            incl_keys = [k for k in keys if len(k)==level]
            spectra = np.array([np.outer(self.nodes[i].classifier,
                                     self.nodes[n].fns[i]) for i in incl_keys]).reshape((-1,
                                                              len(self.nodes[n].classifier),
                                                              len(self.nodes[n].map))).sum(axis=0)
            pref = np.multiply(self.weights, np.multiply(eL.T, spectra).sum(axis=0))
            #print(pref.shape)
            try:
                self.nodes[n].S_update -= np.flatten(scale*np.mean(pref)*self.nodes[n].splitter[0])
            except AttributeError:
                pass
            
    def update_from_level_S_II(self, level, data, scaling_factor = 2):        
        # then update the same-level nodes
        eL = self.remainder_at_level(data, level)
        scale = scaling_factor**level/np.sum(scaling_factor**np.arange(self.get_depth()+1))
        incl_nodes = [n for n in self.nodes if len(n) == level]
        for n in incl_nodes:
            pref = np.multiply(self.weights,
                               np.multiply(eL.T,
                                           -self.nodes[n].map))
            self.nodes[n].S_update -= scale*np.sum(pref, axis=1)/np.maximum(np.sum(self.nodes[n].map), 100)
            
    def update_from_level_S_V(self, data, beta = 0.1, alg='complex', a_len_max=0.01, n_update_points=0,
                             attenuation = 1, levels=(-1,), occs=(), split_var=()):
        # only set up to work on the lowest level
        # Archetypal -analysis update
        # dependence on beta removed via line search
        if self.verbose:
            print("update_from_level_S_V(self, data=", data.shape, 
                ", beta =", beta, ", alg='", alg, "', a_len_max=", a_len_max,
                ", n_update_points=", n_update_points, ", attenuation =", attenuation,
                ", levels=", levels, ", occs=", occs.keys(), ", split_var=", split_var, ")")
        start = time.time()
        if levels[0]==-1:
            levels = (self.get_depth(),)
        #else:
        #    base_level = level
        
            #self.display_level(level)
        
        #incl_classifiers = copy.deepcopy([self.nodes[n].classifier for n in incl_nodes])
        #data_stack = np.vstack([incl_classifiers, data])
        if len(split_var)>0:
            sdata=split_var
        else:
            sdata = data
        if n_update_points>0:
            if len(occs)==0:
                if alg=='complex':
                    self.predict(sdata)
                elif alg=='simple':
                    self.simple_predict(sdata)
                occs = {}
                for n in self.nodes:
                    occs[n] = self.nodes[n].map 
        else:
            if alg=='complex':
                self.predict(sdata)
            elif alg=='simple':
                self.simple_predict(sdata)
        self.get_full_weights()
        
        self.PAA_i += 1
        indices = np.arange(0,len(data))
        for level in levels:
            if self.verbose:
                print("into _V level ", level)
            incl_nodes = [n for n in self.nodes if len(n) == level]
            #eL = self.remainder_at_level(data, level)
            #for pix in data:S
            #if '1' in incl_nodes:
            #    incl_nodes = ['1']
            for n in incl_nodes:
                if self.verbose:
                    print("node", n)
                if n_update_points > 0:
                    om = occs[n]
                    om = om > 0
                    
                    update_pix = np.random.choice(om.sum(), np.minimum(n_update_points, np.sum(om)))
                    #update_pix = rand_sel(om, n_update_points)
                    #ldata = data[update_pix]
                    #print(ldata.shape, "ldata")
                    self.subsamp = indices[om][update_pix]
                    ldata = data[om][update_pix]
                    self.simple_predict(sdata[indices[om][update_pix]])
                    self.get_full_weights()
                else:
                    self.subsamp = []
                    ldata = data
                    self.get_full_weights()

                oldata = ldata #np.concatenate((ldata, self.nodes[n[:-1]].classifier_r.reshape((1,-1))))
                #normalize
                ldata = self.get_ldata(ldata)
                #ldata = np.concatenate((ldata, self.nodes[n[:-1]].classifier.reshape((1,-1))))
                eL = self.remainder_at_level(oldata, level)
                #eL = np.concatenate(eL
                # print("self.weights.shape", self.weights.shape, "self.nodes[n].map.astype(np.float64).shape", self.nodes[n].map.astype(np.float64).shape)
                # print("eL.T.shape", eL.T.shape)

                
                elo_sum = np.sum(np.multiply((self.weights*self.nodes[n].map.astype(np.float64)).T, eL.T), axis=1)
                #denom_2_sum = self.weights * self.nodes[n].map**2
                denom_2_sum = self.weights * self.nodes[n].map.astype(np.float64)**2
                
                #print(n, self.nodes[n].map[:10])
                #print(denom_2_sum[:10])
                #print(self.weights[:10])
                #plt.imshow(denom_2_sum.reshape(self.plot_size))
                denom = np.sum(denom_2_sum)
                #print("nodesum", n, self.nodes[n].map.sum())
                if self.nodes[n].map.sum()==0:
                    next
                else:
                    if self.uncon:
                        ds = elo_sum / denom
                        #plt.plot(self.nodes[n].classifier[:])
                        #plt.plot(self.nodes['0'].classifier[:],'--', label='00')
                        #plt.plot(self.nodes['1'].classifier[:],'-.', label='01')
                        self.nodes[n].classifier[:] += ds[:]/2
                        self.nodes[n].classifier_r[:] = self.nodes[n].classifier[:]
                        self.nodes[n].classifier_n[:] = self.nodes[n].classifier[:]
                        #plt.plot(self.nodes[n].classifier[:])
                        #plt.plot(self.nodes['0'].classifier[:],'--', label='0')
                        #plt.plot(self.nodes['1'].classifier[:],'-.', label='1')
                        #self.nodes[n].classifier_n += ds
                        #self.nodes[n].classifier_r += ds
                        #plt.title(n)
                        #plt.legend()
                        #plt.plot(ds)
                        #plt.show()
                        #print(n, ds.shape)

                    else:
                        lc = self.nodes[n].classifier
                        rem = ldata-lc
                        #rem = np.concatenate((rem, 0*lc.reshape((1, -1))))
                        r2 = (self.nodes[n[:-1]].classifier-lc).flatten()# to throw errors if there is a bug
                        #eXdelta = rem@eL.T
                        #if n_update_points > 0:
                        #    elo_sum = np.sum(np.multiply((self.weights).T, eL.T), axis=1)
                        #    self_int = np.array([self.weights[i]*rem[i]@eL[i] for i in range(len(rem))])
                        #    self_int_denom = np.array([self.weights[i] * self.nodes[n].map[i] for i in range(len(rem))])
                        #else:
                        #elo_sum = np.sum(np.multiply((self.weights*self.nodes[n].map).T, eL.T), axis=1)
                        elo = elo_sum@rem.T
                        elo2 = elo_sum@r2                
                        #norm = np.array([i@i for i in rem])
                        #elo = np.multiply((self.weights*self.nodes[n].map).T,eXdelta)#.T@rem.T
                        #elo_sum = elo.sum#eld = np.array([elo[i,i] for i in range(len(elo))])
                        #if n_update_points > 0:
                        #    denom_2_sum = self.weights * self.nodes[n].map + 1e-16
                        #else:
                        #denom_2_sum = self.weights * self.nodes[n].map**2 #+ 1e-16
                        #denom_adj = self.weights[i] * self.nodes[n].map[i]**2
                        #denom = np.sum(denom_2_sum)
                        norm = np.array([i@i for i in rem]) +1e-16
                        norm_up = r2@r2
                        #print(len(norm),'norm')
                        #slope = np.sbeum((eL.T*self.weights*self.nodes[n].map).T@rem.T, axis=1)
                        if self.aa:
                            self_int = np.array([self.weights[i]*self.nodes[n].map[i]*rem[i]@eL[i] for i in range(len(rem))])
                            self_int_denom = np.array([self.weights[i] * self.nodes[n].map[i]**2 for i in range(len(rem))])
                            #self_int_denom = np.array([self.weights[i] * self.nodes[n].map[i]**2 for i in range(len(rem))])
                            a = (elo - self_int)/((denom - self_int_denom)*norm)#np.sum(elo-np.diag(eld), axis=1)/(denom*norm)
                            a_keep = ((a<1) & (a>0)) #& (slope < 0)
                            a = np.maximum(np.minimum(a, 1),0)
                        else:
                            a = np.ones_like(norm, dtype=np.float32)
                        #a_plus = a > 0
                        #print(elo.shape)
                        #prefactor_a = norm*np.sum(self.nodes[n].map**2*self.weights)
                        total_change = - (denom*norm)/2*a**2 + elo.T*a.T
                        change_up = - (denom*norm_up)/2 + elo2


                        if (~self.aa):#&(self.nodes[n].map.max() < 1):
                            energies = -total_change

                            ##up_energy = -change_up
                            log_num = (self.nodes[n[:-1]].map).sum()
                            aeng = np.average(energies, weights=self.nodes[n].map**2)
                            #lbeta = np.abs(np.log(log_num/(1-self.A_max+1e-64))/(aeng - energies.min()))
                            #print("beta is", lbeta, self.A_max)
                            #energies = energies.tolist()+[0]
                            #energies = np.array(energies)
                            #if self.nodes[n].beta_expectation[-1]==1e-12:
                            #    self.calc_betas(n, energies)
                            #else:
                            #    self.nodes[n].beta_expectation.append(self.nodes[n].beta_expectation[-1]*1.5)
                            #lbeta = self.nodes[n].beta_expectation[-1]
                            #b_energies = -lbeta * energies 
                            #b_energies -= b_energies.max()
                            #probs = prob(energies.astype(np.float128), lbeta)
                            #probs = np.exp(b_energies)
                            #tc_usrt = rand_sel(probs.astype(np.float128), 2)
                            #print("beta is", lbeta, pix)
                            #if pix>=len(a):
                                #selecting to retain the same spectra
                            #    next

                            tc_sorted = np.argsort(total_change)
                            i1 = (self.PAA_i % int(np.sqrt(self.PAA_backcount)))
                            i2 = (self.PAA_i // int(np.sqrt(self.PAA_backcount))) % int(np.sqrt(self.PAA_backcount))
                            i = -i1 * int(np.sqrt(self.PAA_backcount)) - i2 -1
                            pix = tc_sorted[i]
                            classifiers = self.all_classifiers()
                            for c in classifiers:
                                if np.sum((ldata[pix]-c)**2)==0:
                                    i -= 1
                            pix = tc_sorted[i]
                                
                            
                            if self.verbose:
                                print(n, total_change[pix], i)#, [int(i) for i in total_change[tc_sorted][-self.PAA_backcount:]], i)

                            best_sig2 = pix
                            top_sum = 1#a[best_sig2]
                        else:
                            best_sig2 = np.argmax(total_change)
                            if total_change[best_sig2] < 0:
                                next


                        try:
                            top_sum = a[best_sig2]
                        except IndexError:
                            pass
                        #print('1', total_change[best_sig], total_change[best_sig2], slope[best_sig2])
                        #print(n, a)
                        #if slope[best_sig2] < 0:
                        # = self.weights * self.nodes[n].map**2 + 1e-16
                        #print(top_sum, np.sum(denom_2_sum))
                        beta = top_sum #/ np.sum(denom_2_sum)
                        beta = np.minimum(np.maximum(beta,0),1)
                        #beta /= attenuation
                        #print(n, beta, time.time() - start)
                        try:
                            nspec = oldata[best_sig2]
                        except IndexError:
                            pass

                        classifier_r_new = (1-beta)*self.nodes[n].classifier_r + beta*nspec#+ beta*rem[best_sig2]
                        classifier_n_new = (1-beta)*self.nodes[n].classifier_n + beta*ldata[best_sig2]#2lines
                        #classifier_n_new += beta*nspec/np.sqrt(np.sum(nspec**2))
                        #eL += np.outer(self.nodes[n].map, beta*rem[best_sig])
                        self.nodes[n].classifier_r = classifier_r_new
                        self.nodes[n].classifier_n = classifier_n_new
                        if self._use_norm:
                            self.nodes[n].classifier = classifier_n_new
                        else:
                            self.nodes[n].classifier = classifier_r_new
                    #plt.plot(self.nodes['0'].classifier-self.nodes['1'].classifier)
                    #plt.show()


                        #num_2_sum = np.multiply(rem@np.multiply(self.weights, eL.T), self.nodes[n].map)

                        #a[ a < 0 ] = 0
                        #a /= (len(self.weights))#*np.maximum(1,np.sum(self.nodes[n].map))
                        #rem =  np.sum(np.multiply(a, (data_stack-lc).T), axis=1)




                #a *= beta
                #a_mag = np.sum(a)
                #print(beta)
                #if a_mag > a_len_max:
                #    a = (a / a_mag) * a_len_max
                #rem =  np.sum(np.multiply(a, data_stack.T), axis=1)
                #classifier_new = self.nodes[n].classifier*(1-a.sum()) + rem
    
    def calc_betas(self, node, energies):
        btype = 'expectation'
        beta_0 = self.nodes[node].beta_expectation[-1]
        b = beta_update(energies, beta_0)
        if b > 0:
            self.nodes[node].beta_expectation.append(np.mean([b, beta_0]))
        else:
            self.nodes[node].beta_expectation.append(beta_0)
        
        beta_0 = self.nodes[node].beta_probability[-1]
        b = beta_update(energies, beta_0, alg_type='prob')
        if b > 0:
            self.nodes[node].beta_probability.append(np.mean([b, beta_0]))
        else:
            self.nodes[node].beta_probability.append(beta_0)
        
    def grow_network_open(self, image, beta, tol, n_update_pts=1000, obj_record=(),
                          sampling_points=(), scaling_factor=2):
        self.parameter_initialization(image)
        self.initialize_nodes(image)
        self.display_level(1)
        if len(sampling_points) > 0:
            self.nodes[''].map = np.ones(len(sampling_points))
        self.quick_alt_ll(image, beta=beta, n_update_points=n_update_pts,
                                tol=tol, obj_record=obj_record, sampling_points=sampling_points,
                         scaling_factor=scaling_factor)
        
        #self.lowest_alternating(image, beta=beta, n_update_points=n_update_pts,
        #                        tol=tol, obj_record=obj_record, sampling_points=sampling_points)
        self.predict(image)
        while self.check_splitting():
            self.add_another_node_layer(image)
            self.update_spectra(image, self.get_depth(), 'average20')
            self.quick_alt_ll(image, beta=beta, n_update_points=n_update_pts,
                                tol=tol, obj_record=obj_record, sampling_points=sampling_points,
                         scaling_factor=scaling_factor)
            self.predict(image)
            self.display_level(self.get_depth())
            
    def lowest_alternating(self, data, beta=0.1, tol=0.01, n_update_points=1000, sampling_points=(),
                      obj_record=()):
        if len(obj_record)==0:
            obj_record = []
        depth = self.get_depth()
        if len(sampling_points)==0:
            def evaluate():
                self.predict(data)
                eL = self.remainder_at_level(data, depth)
                obj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                return obj
        else:
            def evaluate():
                self.predict(data[sampling_points])
                eL = self.remainder_at_level(data[sampling_points], depth)
                obj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                return obj
        delta = 1
        start = time.time()
        while delta > tol:
            #print("restarting loop")
            obj_orig = evaluate()
            #print("1st eval", time.time()-start)
            delta_II = tol + 1
            obj_sp_orig = obj_orig
            #print("appending", time.time()-start)
            obj_record.append([obj_orig, 0, self.get_depth()])
            local_beta = beta
            while delta_II > tol:
                print("back in loop")
                update_pix = np.random.choice(len(data), n_update_points)
                #self.weights=self.wf(data[update_pix])
                #print("starting lowest s", time.time()-start)
                self.lowest_level_S(data[update_pix],beta=local_beta)
                #print("done lowest s", time.time()-start)
                new_obj = evaluate()
                delta_II = np.abs((obj_sp_orig - new_obj)/obj_sp_orig)
                print(obj_sp_orig, delta_II)
                obj_sp_orig = new_obj
                obj_record.append([new_obj,2, self.get_depth()])
            delta_II = tol+1
            local_beta = beta
            print("out of loop")
            while delta_II > tol:            
                update_pix = np.random.choice(len(data), n_update_points)
                #self.weights=self.wf(data[update_pix])
                self.lowest_spatial(data[update_pix],beta=local_beta)
                new_obj = evaluate()
                delta_II = np.abs((obj_sp_orig - new_obj)/obj_sp_orig)
                print(obj_sp_orig, new_obj,delta_II, local_beta)
                obj_sp_orig = new_obj
                obj_record.append([new_obj, 1, self.get_depth()])

            delta = np.abs((obj_orig - new_obj)/obj_orig)
            print("big step", obj_orig, new_obj, delta)
            
    def quick_alt_ll(self, data, beta=0.1, tol=0.01, n_update_points=200, sampling_points=(),
                      obj_record=(), up_level=0, scaling_factor=2, alg='complex',
                      record_weights=False):
        if len(obj_record)==0:
            obj_record = []
        depth = self.get_depth()
        if len(sampling_points)==0:
            def evaluate():
                if alg=='complex':
                    S=self.predict(data)
                elif alg=='simple':
                    S=self.simple_predict(data)
                obj = 0
                av_max = np.min(np.max(S, axis=1))
                lrec= [sparsity(S), av_max]
                for i in range(1,depth+1):
                    eL = self.remainder_at_level(data, i)
                    lobj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                    obj += (scaling_factor**i)*lobj
                    #print(lobj)
                    lrec.append(lobj)
                return obj, lrec
        else:
            def evaluate():
                if alg=='complex':
                    S=self.predict(data[sampling_points])
                elif alg=='simple':
                    S=self.simple_predict(data[sampling_points])
                obj = 0
                av_max = np.min(np.max(S, axis=1))
                lrec= [sparsity(S), av_max]
                for i in range(1,1+depth):
                    eL = self.remainder_at_level(data[sampling_points], i)
                    lobj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                    obj += (scaling_factor**i)*lobj
                    lrec.append(lobj)
                    #print(lobj)
                return obj, lrec
        delta = tol+1
        obj_orig, o_scores = evaluate()
        while delta > tol:
            #update_pix = np.random.choice(len(data), n_update_points)
            if alg=='simple':
                self.simple_predict(data)
            else:
                self.predict(data)
            prob_map = {}
            for n in self.nodes:
                if len(n) >= 0:
                    prob_map[n] = self.nodes[n].map
                else:
                    prob_map=()
            #print(prob_map.keys())
            self.lowest_level_S(data,beta=beta, alg=alg, n_update_points=n_update_points,
                               occs=prob_map)
            #update_pix = np.random.choice(len(data), n_update_points)
            self.one_step_cyclic(data, lowest=True, n_update_points=n_update_points,
                                prob_map=prob_map)
            
            new_obj, o_scores = evaluate()
            if record_weights:
                self.append_node_record()
                self.score_record.append([o_scores[-1], o_scores[1]])
            delta = np.abs((obj_orig - new_obj)/obj_orig)
            #delta = 0.5*delta + 0.5*halfdelta
            obj_orig = new_obj
            obj_record.append([o_scores[-1],1.5, self.get_depth(), o_scores[0], o_scores[1]])
            print(new_obj,o_scores)
          
    def one_step_cyclic_ll(self, data, n_update_points=0):
        nodes = list(self.nodes.keys())
        depth = self.get_depth()
        level = depth - 1
        for n in nodes:
            if len(n)==level:
                if n+'1' in nodes:
                    self.nodes[n].h_update = 0 
                    self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                              dtype=np.float32) 
                    if n_update_points > 0:
                        update_pix = np.random.choice(len(data), n_update_points)
                        self.node_grad(n, data[update_pix], var='W')
                        self.node_grad(n, data[update_pix], var='h')
                    else:
                        self.node_grad(n, data, var='W')
                        self.node_grad(n, data, var='h')
            
    def quick_alt(self, data, beta=0.1, tol=0.01, n_update_points=200, sampling_points=(),
                  obj_record=(), up_level=0, scaling_factor=2, alg='complex',
                  record_weights=False, split_var=(), both=True, max_iter=1000, only_ends=False,
                  levels=(), A_protection=False):
        if self.verbose:
            print("quick_alt()")
        if len(obj_record)==0:
            obj_record = []
        depth = self.get_depth()
        if len(sampling_points)==0:
            def evaluate():
                if len(split_var)>0:
                    sdata = split_var
                else:
                    sdata = data
                if alg=='complex':
                    S=self.predict(sdata)
                elif alg=='simple':
                    S=self.simple_predict(sdata)
                self.subsamp=[]
                self.get_full_weights()
                obj = 0
                av_max = np.min(np.max(S, axis=1))
                #print(av_max)
                lrec= [sparsity(S), av_max]
                for i in range(1,depth+1):
                    eL = self.remainder_at_level(data, i)
                    lobj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                    obj += (scaling_factor**i)*lobj
                    #print(lobj)
                    
                    lrec.append(lobj)
                return obj, lrec, S
        else:
            def evaluate():
                if len(split_var)>0:
                    sdata = split_var
                else:
                    sdata = data
                if alg=='complex':
                    S=self.predict(sdata[sampling_points])
                elif alg=='simple':
                    S=self.simple_predict(sdata[sampling_points])
                self.subsamp=sampling_points
                self.get_full_weights()
                obj = 0
                av_max = np.min(np.max(S, axis=1))
                #print(av_max)
                lrec= [sparsity(S), av_max]
                for i in range(1,1+depth):
                    eL = self.remainder_at_level(data[sampling_points], i)
                    lobj = (np.sum((eL.T**2), axis=0)*self.weights).mean()
                    obj += (scaling_factor**i)*lobj
                    lrec.append(lobj)
                    #print(lobj)
                return obj, lrec, S
        delta = 1
        obj_orig, o_scores, S = evaluate()
        #print(obj_orig,o_scores)
        if len(split_var)>0:
            sdata = split_var
        else:
            sdata = data
        i = 0 
        while (delta > tol)&(i<max_iter):
            i+=1
            
            if alg=='simple':
                self.simple_predict(sdata)
            else:
                self.predict(sdata)
            self.subsamp=[]
            self.get_full_weights()
            prob_map = {}
            en = self.get_end_nodes()
            if len(sampling_points)<1:
                S_0 = np.zeros((len(en),data.shape[0]))
                S_1 = np.zeros((len(en),data.shape[0]))
            else:
                S_0 = np.zeros((len(en),len(sampling_points)))
                S_1 = np.zeros((len(en),len(sampling_points)))
            
            for n in self.nodes:
                if len(n) >= 0:
                    prob_map[n] = self.nodes[n].map
            #print(prob_map.keys())
            #update_pix = np.random.choice(len(data), np.minimum(len(data), n_update_points))
            #self.one_step(data[update_pix],beta=beta, up_level=up_level, scaling_factor=scaling_factor, alg=alg)
            new_obj, o_scores, S_0[:] = evaluate()
            self.A_max = o_scores[1]
            if not only_ends:
                self.one_step_S(data,beta=beta, up_level=up_level, scaling_factor=scaling_factor, alg=alg,
                           n_update_points = n_update_points, attenuation=1, occs=prob_map, split_var=split_var,
                               levels=levels)
            #update_pix = np.random.choice(len(data), n_update_points)
            #self.one_step_cyclic(data[update_pix], scaling_factor=scaling_factor)
            #print(data.shape, split_var.shape)
            
            #delta = np.abs((obj_orig - new_obj)/obj_orig)
            #delta = 0.5*delta + 0.5*halfdelta
            #obj_orig = new_obj
            #obj_record.append([o_scores[-1],3, self.get_depth(), o_scores[0], o_scores[1]])
            #print(new_obj,o_scores, "after V")
            old_splitters = {n:copy.deepcopy(self.nodes[n].splitter) \
                             for n in self.nodes if len(n) > self.get_depth()}
            self.one_step_cyclic(data, scaling_factor=scaling_factor, n_update_points=n_update_points,
                             prob_map=prob_map, split_var=split_var, both=both, only_ends=only_ends,
                                levels=levels)
            if record_weights:
                self.append_node_record()
                self.score_record.append([o_scores[-1], o_scores[1]])
            new_obj, o_scores, S_1[:] = evaluate()
            delta = np.abs((obj_orig - new_obj)/(obj_orig+new_obj))
            #delta = 0.5*delta + 0.5*halfdelta
            S_delta = np.sum(np.abs(S_1 - S_0)) / np.sum(S_1)
            obj_orig = new_obj
            obj_record.append([o_scores[-1],3, self.get_depth(), S_delta, o_scores[0], o_scores[1]])
            if self.verbose:
                print(new_obj, delta, S_delta, o_scores)
                print((delta > tol),(i<max_iter))
            # 
            
           
            #for n in self.nodes:
            #    self.nodes[n].bonus_boost = 0#(1-Amax)/4#/np.sqrt(Amax)
            #    Amax = self.nodes[n].map.max()
            #    self.nodes[n].sparsity_balance = self.reg*(1-Amax)/Amax
                
            deltas = []
            en = self.get_end_nodes()
            for k in range(len(en)):
                for j in range(k+1,len(en)):
                    x, y = en[k], en[j]
                    deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
            
            #dm = np.min(deltas)
            #size = np.sum(self.weights>0)/len(self.end_nodes)
            #nodes = self.nodes
            #d = self.get_depth()
            #for l in range(1,self.get_depth()+1):
            #    lnodes = [n for n in nodes if len(n)==l]
            #    lnl = len(lnodes)
            #    size = np.sum(self.weights>0)/lnl
            #    for n in lnodes:
            #        nsum = self.nodes[n].map[self.weights>0].astype(np.float32).sum()
            #        self.nodes[n].bonus_boost = np.maximum((2/3-nsum/(size)),0)**2/(d)
                        
            #for n in self.get_end_nodes():
            #    nsum = self.nodes[n].map[self.weights>0].astype(np.float32).sum()
            #    self.nodes[n].bonus_boost = np.maximum((1/2-nsum/(size)),0)**2

            
            #if (o_scores[1]==0):
            #    for n in self.nodes:
            #        if self.nodes[n].map.max()==0:
            #            j=1
            #            while j <= self.get_depth():
            #                try:
            #                    self.nodes[n[:-j]].splitter = [0*self.nodes[n[:-j]].splitter[0],0]
            #                    j = self.get_depth()
            #                except AttributeError:
            #                    j+=1
            #                #old_splitters[n[:-1]]
            
    def load_from_node_record(self, idx):
        for node in self.node_record:
            try:
                self.nodes[node].splitter = (self.node_record[node]['W'][-idx],
                                             self.node_record[node]['h'][-idx])
            except IndexError:
                pass
            try:
                self.nodes[node].classifier = self.node_record[node]['s'][-idx ]
            except:
                pass
            
    def grow_network_closed(self, image, beta, betab, tol, tolb, n_update_pts=1000, scaling_factor=4,
                    obj_record=(), sampling_points=(), alg='complex', record_weights=False):
        self.parameter_initialization(image)
        self.add_another_node_layer_simple(image)
        print(self.nodes)
        #self.initialize_nodes(image)
        self.display_level(1)
        if len(sampling_points) > 0:
            self.nodes[''].map = np.ones(len(sampling_points))
        if alg=='complex':
            self.predict(image)
        elif alg=='simple':
            self.simple_predict(image)
        #eL = self.remainder_at_level(image, 0)    
        #pos = np.min(self.nodes[''].classifier - eL, axis=1) > 0
        #big_err = np.argmax(np.sum(eL[pos]**2, axis=1))
        #args = np.argsort(((image - self.nodes[''].classifier)**2).sum(axis=1))
        #self.nodes[''].splitter = classifiers_2_svm(self.nodes[''].classifier - eL[pos][big_err],
        #                                                           self.nodes[''].classifier + eL[pos][big_err])
        #self.nodes['0'].classifier = self.nodes[''].classifier
        #self.nodes['1'].classifier = self.nodes[''].classifier
        self.quick_alt_ll(image, beta=beta/100, n_update_points=n_update_pts, tol=tol,
                     obj_record=obj_record, sampling_points=sampling_points, alg=alg,
                         record_weights=record_weights)
        if alg=='complex':
            self.predict(image)
        elif alg=='simple':
            self.simple_predict(image)
        self.display_level(1)
        
    
        while self.check_splitting():
            eL = self.remainder_at_level(image, self.get_depth())
            self.add_another_node_layer_simple(image)
            self.display_level(self.get_depth())
            #self.update_spectra(image, self.get_depth(), 'average2')
            #for n in self.nodes:
            #    if len(n) == (self.get_depth()-1):
            #        if n + '1' in self.nodes:
            #            pos = np.min(self.nodes[n].classifier - eL, axis=1) > 0
            #            big_err = np.argmax(np.sum(eL[pos]**2, axis=1))
            #            self.nodes[n].splitter = classifiers_2_svm(self.nodes[n].classifier - 2*eL[pos][big_err],
            #                                                   self.nodes[n].classifier + 2*eL[pos][big_err])
            #    if len(n) == (self.get_depth()):
            #        self.nodes[n].classifier = self.nodes[n[:-1]].classifier
            if alg=='complex':
                self.predict(image)
            elif alg=='simple':
                self.simple_predict(image)
            #self.update_from_level_S_V(image, alg='simple', attenuation=2)
            print("starting quick ll")
            #self.quick_alt_ll(image, beta=beta/(2**(self.get_depth()-1)), tol=tol/(2**(self.get_depth()-1)),
            #              n_update_points=n_update_pts,
            #              obj_record=obj_record, sampling_points=sampling_points, alg=alg,
            #                 record_weights=record_weights)
            print("ending quick ll")
            old_obj = obj_record[-1][0]
            new_obj = -1
            delta = np.abs(old_obj - new_obj)/old_obj
            new_obj = old_obj
            delta = 1 
            print(delta)
            while delta > tol:
                print("starting full")
                self.quick_alt( image, beta=betab/(2**(self.get_depth()-1)), tol=tolb/(2**(self.get_depth()-1)),
                      n_update_points=n_update_pts,
                            sampling_points=sampling_points, obj_record=obj_record, scaling_factor=scaling_factor,
                              alg=alg, record_weights=record_weights)
                print("starting quick ll")
                #self.quick_alt_ll( image, beta=beta/(1.5**(self.get_depth()-1)), tol=tol/(2**(self.get_depth()-1)),
                #               n_update_points=n_update_pts,
                #               obj_record=obj_record, sampling_points=sampling_points, alg=alg,
                #                 record_weights=record_weights)
                old_obj = new_obj
                new_obj = obj_record[-1][0]
                #delta = np.abs(old_obj - new_obj)/old_obj
                delta = np.abs(old_obj - new_obj)/old_obj#obj_record[-1][-1]
                #print(np.abs(old_obj - new_obj)/old_obj)
            
            if alg=='complex':
                self.predict(image)
            elif alg=='simple':
                self.simple_predict(image)
            self.display_level(self.get_depth())

    def stablize_network(self, data, tol, n_update_points, alg='simple'):
        self.simple_predict(data)
        err = self.remainder_at_level(data, self.get_depth())
        p_err = np.mean(np.sum(err**2, axis=-1))
        var = 1
        while var > tol:
            print(p_err)
            self.one_step_S(data, beta=1, alg=alg, n_update_points=n_update_points, attenuation=1)
            self.simple_predict(data)
            err = self.remainder_at_level(data, self.get_depth())
            ope = p_err
            p_err = np.mean(np.sum(err**2, axis=-1))
            var = np.abs(ope-p_err)/p_err

    def grow_network_single_nodes(self, image, tol=(0.1), n_update_pts=(1000), scaling_factor=2,
                    obj_record=(), sampling_points=(), alg='simple', record_weights=False,
                                 use_norm=True, clean=False, A_tol = 0.0, split_var=(), saturation = ()):
        self.start=time.time()
        self.use_norm(use_norm)
        self.n_update_pts = n_update_pts[0]
        self.training='grow_network_single'
        print("finding neighbors")
        #neighbors_finder = nei.NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(image)
        #__, n_index = neighbors_finder.kneighbors(image)
        self.neighbors = quick_nn(image.reshape(self.plot_size + (-1,)), k_size=1).flatten()
        print("neighbors found") 
        self.set_neighbor_weights(image)
        if len(saturation)==len(image):
            self.full_weights[saturation] = 0
        #self.full_weights *= 0
        #self.full_weights += 1
        self.parameter_initialization(image)
        
        #print(self.nodes)
        self.add_single_node(image, n_update_points = n_update_pts[0])
        #if alg=='complex':
        #    self.predict(image)
        if len(split_var)>0:
            self.simple_predict(split_var)
        else:
            self.simple_predict(image)
        self.display_level(1)
        if len(sampling_points) > 0:
            self.nodes[''].map = np.ones(len(sampling_points))

        print("made it 1")
        self.exempt_node = ''
        self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=A_tol, split_var=split_var)
        self.simple_predict(image)
        self.display_level(1)
        while self.check_single_splitting():
            
            #self.simple_predict(image)
            #self.binarize_lmdas()
            #self.lmda_2_map()
            #self.display_level(self.get_depth())
            #self.update_from_level_S_V(image, beta = 0, alg='simple', a_len_max=0.01, n_update_points=0,
            #                 attenuation = 1, levels=(-1,), occs=(), split_var=image)
            #self.rescale_all_nodes(self.mixed_pix, image)
            #self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[0], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=A_tol, both=False)
            #self.simple_predict(image)
            #self.display_level(self.get_depth())
            #eL = self.remainder_at_level(image, self.get_depth())
            #self.get_full_weights()
            #en = self.get_end_nodes()
            #deltas = []
            #for i in range(len(en)):
            #    for j in range(i+1,len(en)):
            #        x, y = en[i], en[j]
            #        deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
            #self.reg = np.min(deltas)/4
            #self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                     A_protection=False)
            if alg=='complex':
                self.predict(image)
            elif alg=='simple':
                self.simple_predict(image)
            self.display_level(self.get_depth())
            
            
            #o_reg = self.reg
            #for l in range(self.get_depth()-1, -1,-1):
            #    print("sparsifying", l)
            #    #assess how much sparsity is needed:
            #    deltas = []
            #    en = [n for n in self.nodes if len(n)==(l+1)]
            #    for i in range(len(en)):
            #        for j in range(i+1,len(en)):
            #            x, y = en[i], en[j]
            #            deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
            
            #    self.reg = np.min(deltas) + o_reg
            #   self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[1], 
            #                         scaling_factor=scaling_factor, sampling_points=np.arange(image.shape[0]),
            #                         alg=alg, obj_record=obj_record, A_tol=0.5)
            #self.reg = o_reg
            
            print("made it here")
            #self.simple_predict(image)
            #self.binarize_lmdas()
            #self.lmda_2_map()
            #self.display_level(self.get_depth())
            #occs={n:self.nodes[n].map for n in self.nodes}
            #self.update_from_level_S_V(image, beta = 0, alg='simple', a_len_max=0.01, n_update_points=len(image),
            #                 attenuation = 1, levels=(-1,), occs=occs, split_var=image)
            self.simple_predict(image)
            for node in self.nodes:
                self.nodes[node].pmap = copy.deepcopy(self.nodes[node].map)
            self.binarize_lmdas()
            self.lmda_2_map()
            for node in self.nodes:
                self.nodes[node].lmap = copy.deepcopy(self.nodes[node].map)
                self.nodes[node].map[:] = self.nodes[node].pmap
                del self.nodes[node].pmap
            
            
            self.simple_predict(image)
            self.get_full_weights()
            self.add_single_node(image,n_update_points=n_update_pts[0])
            if alg=='complex':
                self.predict(image)
            elif alg=='simple':
                self.simple_predict(image)
            
            
            en = self.get_end_nodes()
            deltas = []
            for i in range(len(en)):
                for j in range(i+1,len(en)):
                    x, y = en[i], en[j]
                    deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))

            #self.reg = -np.min(deltas)
            #self.switch_training(image, beta=0, tol=1e-4, n_update_points=n_update_pts[1], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                     A_protection=False)
            md = np.min(deltas)

            
            k = 0
            A_tol=1
            n_steps = 5
            mdstep = md/2/n_steps
            while (k<n_steps)&(A_tol>0.5):
                #self.untangle_endmembers(image)
                en = self.get_end_nodes()
                deltas = []
                for i in range(len(en)):
                    for j in range(i+1,len(en)):
                        x, y = en[i], en[j]
                        deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))

                #self.reg = -np.min(deltas)
                #self.switch_training(image, beta=0, tol=1e-4, n_update_points=n_update_pts[1], 
                #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
                #                     A_protection=False)
                md = np.min(deltas)

                self.simple_predict(image)
                self.display_level(self.get_depth())
                self.reg = -k*mdstep
                self.switch_training(image, beta=0, tol=1e-12, n_update_points=n_update_pts[-1], 
                                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=2,
                                     A_protection=False)
                k+=1
                if alg=='complex':
                    S = self.predict(image)
                elif alg=='simple':
                    S = self.simple_predict(image)
                self.display_level(self.get_depth())

                A_tol = np.min(np.max(S, axis=1))
                k += 1




                #self.rescale_all_nodes(1-self.mixed_pix, image, less_than=False)
                #if alg=='complex':
                #    S = self.predict(image)
                #elif alg=='simple':
                #    S = self.simple_predict(image)
                #self.display_level(self.get_depth())
                # 
                #A_tol = np.max(S)
                #print(A_tol)
            
            #self.display_level(self.get_depth())
            
            #self.reg = 0
            ##self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
            #                         scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                         alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                         A_protection=False)
            
            
            k=1
            A_tol = 0
            mpmp=1
            mdstep = 0.005
            while (k<1)|(mpmp>0.5):
                #self.untangle_endmembers(image)
                if alg=='complex':
                    S = self.predict(image)
                elif alg=='simple':
                    S = self.simple_predict(image)
                self.display_level(self.get_depth())
                mpmp = self.mpmp()
                

                A_tol = np.min(np.max(S, axis=1))
                k += 1
                if A_tol > 0.5:
                    self.reg += mdstep
                print(self.reg, mpmp)
                self.switch_training(image, beta=0, tol=1e-6, n_update_points=n_update_pts[-1], 
                                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=2,
                                     A_protection=False)


                
                
                #self.rescale_all_nodes(1-self.mixed_pix, image, less_than=False)
                
            
            #self.reg = 0
            self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
                                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
                                     A_protection=False)
            self.reg = 0
            self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
                                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
                                     A_protection=False)
            #self.rescale_all_nodes(0.1, image, less_than=False)
            
            #self.binarize_lmdas()
            #self.lmda_2_map()
            #self.update_from_level_S_V(image, beta = 0, alg='simple', a_len_max=0.01, n_update_points=0,
            #                 attenuation = 1, levels=(-1,), occs=(), split_var=image)
            #self.fix_end_classifiers(self.get_ldata(image), image)
            
            #if alg=='complex':
            #    self.predict(image)
            #elif alg=='simple':
            #    self.simple_predict(image)
            #self.display_level(self.get_depth())
            #o_reg = self.reg
            
            #for l in range(0,self.get_depth()):
            #    print("de sparsifying", l)
            #    deltas = []
            #    en = [n for n in self.nodes if len(n)==(l+1)]
            #    for i in range(len(en)):
            #        for j in range(i+1,len(en)):
            #            x, y = en[i], en[j]
            #            deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
            
            #    self.reg = np.min(deltas) + o_reg
            #    self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[1], 
            #                         scaling_factor=scaling_factor, sampling_points=np.arange(image.shape[0]),
            #                         alg=alg, obj_record=obj_record, A_tol=1e-2, only_ends=True,
            #                         A_protection=False)
            #    self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[1], 
            #                         scaling_factor=scaling_factor, sampling_points=np.arange(image.shape[0]),
            #                         alg=alg, obj_record=obj_record, A_tol=1e-2,
            #                        A_protection=False)
            
            
           
            
            

            #en = self.get_end_nodes()
            #deltas = []
            #for i in range(len(en)):
            #    for j in range(i+1,len(en)):
            #        x, y = en[i], en[j]
            #        deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
            #self.reg = -np.min(deltas)/4
            #self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                     A_protection=False)
            
            #if alg=='complex':
            #    self.predict(image)
            #elif alg=='simple':
            #    self.simple_predict(image)
            #self.display_level(self.get_depth())
            
            #self.reg = 0
            #self.switch_training(image, beta=0, tol=tol[1], n_update_points=n_update_pts[-1], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                     A_protection=False)
            
            
            #self.blur_all_nodes(image)
            
            
            #self.switch_training(image, beta=0, tol=tol[0], n_update_points=n_update_pts[0], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_Ftol=A_tol)  
         
            
            
            
        #self.full_weights *= 0
        #self.full_weights += 1
        #self.switch_training(image, beta=0, tol=tol[-1], n_update_points=n_update_pts[-1], 
        #                             scaling_factor=scaling_factor, sampling_points=sampling_points,
        #                             alg=alg, obj_record=obj_record, A_tol=A_tol)  
        en = self.get_end_nodes()
        deltas = []
        for i in range(len(en)):
            for j in range(i+1,len(en)):
                x, y = en[i], en[j]
                deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))

        #self.reg = -np.min(deltas)
        #self.switch_training(image, beta=0, tol=1e-4, n_update_points=n_update_pts[1], 
        #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
        #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
        #                     A_protection=False)
        md = np.min(deltas)

        self.use_bonus_boost=False
        k = 0
        A_tol=0
        n_steps = 5
        mdstep = md/2/n_steps
        for k in range(n_steps):
            #self.untangle_endmembers(image)
            en = self.get_end_nodes()
            deltas = []
            for i in range(len(en)):
                for j in range(i+1,len(en)):
                    x, y = en[i], en[j]
                    deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))

            #self.reg = -np.min(deltas)
            #self.switch_training(image, beta=0, tol=1e-4, n_update_points=n_update_pts[1], 
            #                     scaling_factor=scaling_factor, sampling_points=sampling_points,
            #                     alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=self.max_iter,
            #                     A_protection=False)
            md = np.min(deltas)

            self.simple_predict(image)
            self.display_level(self.get_depth())
            self.reg = -k*mdstep
            self.switch_training(image, beta=0, tol=1e-12, n_update_points=n_update_pts[-1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=1,
                                 A_protection=False)
        
        en = self.get_end_nodes()
        deltas=[]
        for i in range(len(en)):
            for j in range(i+1,len(en)):
                x, y = en[i], en[j]
                deltas.append(np.sum((self.nodes[x].classifier-self.nodes[y].classifier)**2))
        print("min deltas is", np.min(deltas))
        
        k=1
        A_tol = 0
        spars = 0
        mdstep = 0.005
        mpmp = 0
        while ((k<1)|(A_tol<1))|(mpmp>0.1):
            #self.untangle_endmembers(image)
            if alg=='complex':
                S = self.predict(image)
            elif alg=='simple':
                S = self.simple_predict(image)
            self.display_level(self.get_depth())

            A_tol = np.min(np.max(S, axis=1))
            spars = sparsity(S)
            mpmp = self.mpmp()
            k += 1
            if A_tol > 0.5:
                self.reg += mdstep
            self.switch_training(image, beta=0, tol=1e-6, n_update_points=n_update_pts[-1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=2,
                                 A_protection=False)



            print(self.reg, mpmp)
            
        
        k=0
        A_tol = 0
        spars = 0
        while ((A_tol==1))&(self.reg>0):
            #self.untangle_endmembers(image)
            if alg=='complex':
                S = self.predict(image)
            elif alg=='simple':
                S = self.simple_predict(image)
            self.display_level(self.get_depth())

            A_tol = np.min(np.max(S, axis=1))
            spars = sparsity(S)
            mpmp = self.mpmp()
            k += 1
            self.reg -= mdstep
            self.switch_training(image, beta=0, tol=1e-6, n_update_points=n_update_pts[-1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=1e-4, max_iter=2,
                                 A_protection=False)



            print("declining",self.reg, mpmp)
        
        
        
        
        
        
        
        
        
        
        
        self.reg=0
        self.switch_training(image, beta=0, tol=tol[-1], n_update_points=n_update_pts[-1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=1e-6)
        
        
        
        self.simple_predict(image)
        self.display_level(self.get_depth())
        
        if clean:
            self.aa =True
            
            #self.use_norm(False)
            #self.stablize_network(image, tol=tol, n_update_points=n_update_pts)
            #not that, because AA is not as sparse, A_tol is dropped
            self.switch_training(image, beta=0, tol=tol[-1], n_update_points=n_update_pts[-1], 
                                 scaling_factor=scaling_factor, sampling_points=sampling_points,
                                 alg=alg, obj_record=obj_record, A_tol=1e-6)
            
        self.end=time.time()
        self.training += ";" + str(int(self.end-self.start))

    def switch_training(self, image, beta, tol, sw_tol=1e-1, n_update_points=1000, scaling_factor=4,
                        obj_record=(), sampling_points=(), alg='complex', A_tol=0, 
                        split_var=(), both=True, only_ends=False, A_protection=False,
                        max_iter=1000):
        #self.quick_alt_ll( image, beta=beta, tol=tol,
        #                       n_update_points=n_update_points,
        #                       obj_record=obj_record, sampling_points=sampling_points, alg=alg)
        old_obj = 1
        new_obj = 0
        delta = 1
        A = 0 
        #print(split_var)
        if len(obj_record)==0:
            obj_record = [[1,0,0,0]]
        while (A < A_tol): #(delta > sw_tol) |
            #print("inside")
            
            self.quick_alt( image, beta=beta, tol=tol,
              n_update_points=n_update_points,
                    sampling_points=sampling_points, obj_record=obj_record,
                       scaling_factor=scaling_factor, alg=alg, split_var=split_var,
                      both = both, only_ends=only_ends, A_protection=A_protection,
                          max_iter=max_iter)
            #self.quick_alt_ll( image, beta=beta, tol=tol,
            #               n_update_points=n_update_points,
            #               obj_record=obj_record, sampling_points=sampling_points, alg=alg)
            
            #print(obj_record[-1])
            try:
                if new_obj > 0:
                    old_obj = new_obj
                new_obj = obj_record[-1][0]
                A = obj_record[-1][-1]
            except IndexError:
                pass
            
            #delta = (np.abs(old_obj - new_obj)/old_obj)
            #print("st", delta, old_obj, new_obj)
            
            #if (A_tol == 1)&(delta < tol):
            #    if A < A_tol:
            #        self.fix_end_classifiers(self.get_ldata(image), image)
           #print(da, np.abs(old_obj - new_obj)/old_obj,
               #       (np.abs(old_obj - new_obj)/old_obj > tol), (da > A_tol))
            
    def intermediate_lambda_product(self, node_a, node_b):
        if node_b[:len(node_a)]==node_a:
            lamb_num = len(node_b) - len(node_a)
            lamb = np.ones_like(self.nodes[node_a].lmda)
            i = 0
            while i < lamb_num:
                lamb[:] *= self.nodes[node_b[:(len(node_a)+i+1)]].lmda[:]
                i+=1
            return lamb

        else:
            print("not a valid lambda product")
            return -1
        
    def aggregate_node_at_level(self, node, level, data, vtype='s'):
        top_level = len(node)
        
        nodes_in_level = [i for i in self.nodes if len(i)==level]
        nodes_in_level.sort()
        if vtype=='s':
            out = np.zeros_like(data)
        elif vtype=='a':
            out = np.zeros((data.shape[0], len(nodes_in_level)))
        #print("nil", nodes_in_level)

        for i, snode in enumerate(nodes_in_level):
            if snode[:(top_level)]==node:
                #switch out for lambda
                #plt.plot(self.nodes[snode].classifier)
                
                if snode[top_level]=='0':
                    #print('snowd',snode)
                    loc_lambda = self.intermediate_lambda_product(node + '0', snode)
                    if vtype=='s':
                        out += np.outer(loc_lambda, self.nodes[snode].classifier)
                    elif vtype=='a':
                        out[:,i] += loc_lambda * self.nodes[snode].map
                    #print("locl", loc_lambda.max(), self.nodes[snode].classifier.max(), out.max())
                elif snode[top_level]=='1':
                    #print('snowd',snode)
                    loc_lambda = self.intermediate_lambda_product(node + '1', snode)
                    #print("locl", loc_lambda.max(), self.nodes[snode].classifier.max())
                    if vtype=='s':
                        out -= np.outer(loc_lambda, self.nodes[snode].classifier)
                    elif vtype=='a':
                        out[:,i] -= loc_lambda * self.nodes[snode].map
        #plt.show()
        
        #print("outmax", out.max())

        return out
    
    def node_grad(self, node, data, scaling_factor=2, var='W', metropolis=False,
                 split_var=(), only_ends = False):
        if len(split_var)>0:
            S=self.simple_predict(split_var.astype(np.float64))
            sdata = split_var
        else:
            S=self.simple_predict(data.astype(np.float64))
            sdata = data
            
        mu = self.nodes[node].mu
        if self.use_bsp:
            if self.both_sides_pure(node):
                r = self.reg
            else:
                r = 0
        else:
            r = self.reg
        
        L = len(self.nodes[node].splitter[0])
        #if var=='W':
        #    try:
        #        r*= split_var.shape[1]
        #    except AttributeError:
        #        r*= data.shape[1]
        #if node == self.exempt_node:
        #    r = self.reg * self.exemption_rate
        #else:
        #    r = self.reg
        self.get_full_weights()
        #self.populate_fns(data, alg='simple')
        depth = self.get_depth()
        eL = self.remainder_at_level(data, depth)
        o_err = np.sum(np.multiply((eL**2).T, self.weights), axis=0).astype(np.float64).mean() - r * np.sum(S**2, axis=0).astype(np.float64).mean()
        
        if self.only_ends:
            start_layer = depth
        else:
            start_layer = len(node) + 1
        #print("start layer", node, start_layer)
        
        
        factors = 1.0 * scaling_factor**np.arange(depth + 1)
        factors /= np.sum(factors)
        #print("lmda", self.nodes[node+'0'].lmda.dtype)
        init_incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
        init_incl1 = np.append(init_incl, [True])
        
        ldata = self.get_ldata(data)
        
        
        if var=='W':
            num = np.zeros(sdata.shape)
            na = np.zeros(sdata.shape)
            nb = np.zeros(sdata.shape)
            #num2 = np.zeros(sdata.shape)
            for i in range(start_layer,depth+1):
                eL = self.remainder_at_level(data, i)
                out = self.aggregate_node_at_level(node, i, data)
                out_reg = self.aggregate_node_at_level(node, i, sdata, vtype='a')
                if metropolis:
                    werr = np.sum(np.multiply(eL, out), axis=1)
                else:
                    #werr = np.sum(np.multiply(eL, out), axis=1)*self.nodes[node].map
                    nodes_level_i = [n for n in self.nodes if len(n)==i]
                    nodes_level_i.sort()
                    A = np.array([self.nodes[n].map for n in nodes_level_i]).T
                    #is_exempt = np.array([self.exempt_node in i for i in nodes_level_i])
                    #A *= 1 #+ 2 * is_exempt * (1 - self.exemption_rate)
                    werr = np.sum(np.multiply(eL, out), axis=1)
                    na += factors[i]*np.multiply(sdata.T,np.multiply(werr*self.nodes[node].map, self.weights)).T
                    Areg = np.sum(np.multiply(A, out_reg), axis=1)
                    nb += factors[i]*np.multiply(sdata.T,np.multiply(Areg*self.nodes[node].map, self.weights)).T
                    werr += r*Areg
                    if self.use_bonus_boost:
                        bonus_boosts = np.array([self.nodes[n].bonus_boost for n in nodes_level_i])
                        werr += np.sum(np.multiply(bonus_boosts, out_reg), axis=1)#*Areg #*A
                        r_balance = np.array([self.nodes[n].sparsity_balance for n in nodes_level_i])
                        werr += np.sum(np.multiply(r_balance*A, out_reg), axis=1)
                    werr *= self.nodes[node].map
                werr_w = np.multiply(self.weights, werr)
                #print(node, werr_w.shape, sdata.shape, num.shape)
                num += factors[i]*np.multiply(sdata.T, werr_w).T
            l2_dir = 2*self.nodes[node].splitter[0].reshape(1,-1)
            l2_mean = 2*np.sum(self.nodes[node].splitter[0])*np.ones((1,L))/L
            num = np.append(num, mu*len(num)*(l2_dir + l2_mean), axis=0)
            
                
            #plt.plot(num[-1])
            
               
            self.nodes[node].W_update = -np.mean(num[init_incl1], axis=0)
            #plt.plot(-np.mean(na[init_incl], axis=0))
            #plt.plot(-np.mean(num[init_incl1], axis=0))
            #plt.show()
            #fig, ax = plt.subplots()
            #ax.plot(-np.mean(num[init_incl1][:-1], axis=0))
            #ax.plot(self.nodes[node].W_update)
            #ax0 = ax.twinx()
            #ax0.plot(self.nodes[node].splitter[0], '--', color='black')
            #plt.show()


        #print("wu", self.nodes[node].W_update)
        if var=='W':
            #print(self.nodes[node].W_update)
            xw = sdata@self.nodes[node].W_update/2
            #print(xw)
            if xw.any()==0:
                # TODO: this was enabled and plotted a lot of useless information
                # plt.plot(self.nodes[node].W_update)
                # plt.show()
                pass
        elif var=='h':
            xw = 0.5
            
        #print(node, var, np.mean(np.abs(xw)),'xw')

        num = np.zeros(data.shape[0])
        denom = np.zeros(data.shape[0])

        factors = 1.0 * scaling_factor**np.arange(depth + 1)
        factors /= np.sum(factors)
        

        for i in range(start_layer, depth + 1):#range(len(node)+1,depth+1):
            eL = self.remainder_at_level(data, i)
            out = self.aggregate_node_at_level(node, i, data)
            out_reg = self.aggregate_node_at_level(node, i, data, vtype='a')

            nodes_level_i = [n for n in self.nodes if len(n)==i]
            nodes_level_i.sort()
            A = np.array([self.nodes[n].map for n in nodes_level_i]).T
            werr = np.sum(np.multiply(eL, out), axis=1)
            Areg = np.sum(np.multiply(A, out_reg), axis=1)
            werr += r*Areg #+
            if self.use_bonus_boost:
                bonus_boosts = np.array([self.nodes[n].bonus_boost for n in nodes_level_i])
                werr += np.sum(np.multiply(bonus_boosts, out_reg), axis=1)
                r_balance = np.array([self.nodes[n].sparsity_balance for n in nodes_level_i])
                werr += np.sum(np.multiply(r_balance*A, out_reg), axis=1)

            werr *= self.nodes[node].map
            werr_w = np.multiply(np.multiply(self.weights, werr), xw)
            
            d_sum = np.sum(out**2, axis=1) - r*np.sum(out_reg**2, axis=1) # - r*np.sum(out_reg**2, axis=1)
            #print(np.sum(out[init_incl]**2),np.sum(out_reg[init_incl]**2)) 
            if self.use_bonus_boost:
                d_sum -= np.sum((np.multiply(r_balance, out_reg)*out_reg), axis=1)
            nl = np.multiply(self.weights, np.multiply(d_sum,  xw**2))
            if metropolis:
                num += factors[i]*werr_w#*self.nodes[node].map
                denom += factors[i]*nl*self.nodes[node].map
            else:    
                num += factors[i]*werr_w*self.nodes[node].map
                denom += factors[i]*nl*self.nodes[node].map**2
        #print(node, num.shape, denom.shape)
        if (var=='W'):
            l2_dir = 2*np.dot(self.nodes[node].W_update, self.nodes[node].splitter[0])
            l2_mean = 2*np.sum(self.nodes[node].splitter[0])*np.sum(self.nodes[node].W_update)
            num = np.append(num, [mu*len(num)*(l2_dir + l2_mean)], axis=0)
            l2_dir_D = np.sum(self.nodes[node].W_update**2)
            l2_mean_D = np.sum(self.nodes[node].W_update)**2
            denom = np.append(denom, [mu*len(num)*(l2_dir_D + l2_mean_D)], axis=0)#removed -
        else:
            num = np.append(num, [2*mu*self.nodes[node].splitter[1]*len(num)*len(self.nodes[node].splitter[0])], axis=0)
            denom = np.append(denom, [mu*len(num)*len(self.nodes[node].splitter[0])])
        
        #denom /= self.a_speed

        
        if np.sum(init_incl1)==0:
            return -1
        pl = (sdata@self.nodes[node].splitter[0] + self.nodes[node].splitter[1] +1)/2
        #print(pl.dtype)
        int_level1 = (1-pl)/(xw)
        int_level0 = (-pl)/(xw)
        #if var == 'h':
        #    int_level1 *= -1
        #    int_level0 *= -1
        intercepts = np.minimum(int_level1, int_level0)
        beta = -np.sum(num[init_incl1])/np.sum(denom[init_incl1])
        #print(int_level0.dtype, int_level1.dtype, int_level0[np.argmin(np.abs(int_level0))],
        # #     int_level1[np.argmin(np.abs(int_level1))])
        
        
        if var=='h':
            beta_flag=False
            if beta < 0:
                beta = -beta
                int_level0 = -int_level0
                int_level1 = -int_level1
                num = -num
                beta_flag = True
        #assert beta > 0 , "gradient in wrong direction"
       
                
        #print(int_level0.dtype, int_level1.dtype, int_level0[np.argmin(np.abs(int_level0))],
        #      int_level1[np.argmin(np.abs(int_level1))])
        int_level0[int_level0<1e-64]=10**16
        int_level1[int_level1<1e-64]=10**16
       
        intercepts = np.minimum(int_level1, int_level0)
        intercepts = np.append(intercepts, [10**17])
        
        if self.verbose:
            # TODO: this was enabled and printed a lot of useless information
            # print('veta,', var, beta, intercepts.min(), np.sum(init_incl),np.sum(num[init_incl1]), np.sum(denom[init_incl1]))
            pass
        
        
        if beta > intercepts.min():
            #beta *=0.95
            #beta_0 = beta
            if self.sparsifying==False:
                beta = find_opt_beta(intercepts, init_incl1, num, denom, title=node+var)
            else:
                beta = find_opt_beta(intercepts, init_incl1, num, denom, target = 0.5)
            #betax = np.argmin([np.abs(beta_0), np.abs(beta)])
            #beta = [beta_0, beta][betax]
            
            #if beta > beta_0:
            #    beta = beta_0
            if self.verbose:
                # TODO: this was enabled and printed a lot of useless information
                # print('beta,', beta) 
                pass
            if beta < intercepts.min():
                beta = intercepts.min() - 1e-16
        if beta > 10**10:
            beta = 0#beta_0
        #beta=beta_0


        if var=='W':    
            if beta > 0:
                beta=beta
            else:
                beta = 0
        elif var=='h':
            if np.isnan(beta):
                beta=0
            if beta_flag:
                beta=-beta
        #print("beta", beta)
        #accelerator = 1#.5
        beta *= self.a_speed
        beta_0 = beta 
            
        old_splitter = copy.deepcopy(self.nodes[node].splitter)
        if var=='W':
            splitter = beta*self.nodes[node].W_update + old_splitter[0], old_splitter[1]
        elif var=='h':
            splitter = old_splitter[0], beta + old_splitter[1]
        self.nodes[node].splitter = splitter
        
            
        S=self.simple_predict(sdata.astype(np.float64))
        #print(node, self.node_clarity(node), "node clarity")
        self.get_full_weights()
        eL = self.remainder_at_level(data, depth)
        #print(node, var, ' include ', np.sum(init_incl))
        n_err = np.sum(np.multiply((eL**2).T, self.weights), axis=0).astype(np.float64).mean() - r * np.sum(S**2, axis=0).astype(np.float64).mean()
        n_incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
        if self.verbose:
            # TODO: this was enabled and printed a lot of useless information
            # print('err', var, o_err, n_err, var)
            pass
        if False:#n_err > o_err:
            betas = [0]
            scores = [o_err]
            include = [np.sum(init_incl)]
            
            
            splits = np.arange(-10,15)*0.1
            print(splits)
            #if var=='W':
            #    plt.plot(splitter[0])
            #    plt.plot(old_splitter[0])
            for i, j in enumerate(splits):
                beta_X = beta*j
                self.nodes[node].splitter = old_splitter
                if var=='W':
                    splitter = old_splitter[0] + beta_X*self.nodes[node].W_update, old_splitter[1]
                if var=='h':
                    splitter = old_splitter[0], old_splitter[1] + beta_X
                #if var=='W':
                #    plt.plot(splitter[0])
                self.nodes[node].splitter = splitter
                
                S = self.simple_predict(sdata.astype(np.float64))
                self.get_full_weights()
                eL = self.remainder_at_level(data, depth)
                err = np.sum(np.multiply((eL**2).T.astype(np.float64), self.weights), axis=0).mean() - r * np.sum(S.astype(np.float64)**2, axis=0).mean()
                incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
                betas.append(beta_X)
                scores.append(err)
                include.append(np.sum(incl))
            #if var=='W':
            #    plt.show()
            betas.append(beta)
            scores.append(n_err)
            include.append(np.sum(n_incl))
            #print(betas, scores)
            fig = plt.figure()
            ax = fig.add_subplot(111)    
            ax.plot(betas, scores,'.-')
            ax2 = ax.twinx()
            ax2.plot(betas, include, '.-',color='green')
            ax.set_title(node + " " + var + " r is " + str(r))
            plt.show()
            beta_min = np.argmin(scores)
            if var=='W':
                splitter = beta_0*self.nodes[node].W_update + old_splitter[0], old_splitter[1]
                #splitter = betas[beta_min]*self.nodes[node].W_update + old_splitter[0], old_splitter[1]
            elif var=='h':
                splitter = old_splitter[0], beta_0 + old_splitter[1]
                #splitter = old_splitter[0], betas[beta_min] + old_splitter[1]
            self.nodes[node].splitter = splitter
            S=self.simple_predict(sdata)
            #print(node, self.node_clarity(node), "node clarity")
            self.get_full_weights()
            eL = self.remainder_at_level(data, depth)
            #print(node, var, ' include ', np.sum(init_incl))
            nn_err = np.sum(np.multiply((eL**2).T, self.weights), axis=0).mean() - r * np.sum(S**2, axis=0).mean()
            print('err', var, o_err, n_err, nn_err, scores[beta_min], 'beta', betas[beta_min], beta)
            #assert o_err - n_err > 1e-5
        
        map0 = np.sum(self.nodes[node+'0'].map)==0
        map1 = np.sum(self.nodes[node+'1'].map)==0
        init_incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
        if map0|map1:
            if self.verbose:
                print("back to old splitter", node, var)
            self.nodes[node].splitter = old_splitter
            
        #plt.imshow(self.nodes[node+'0'].lmda.reshape(self.plot_size))
        #plt.colorbar()
        #plt.show()
        #self.display_level(1)
        #if o_err<n_err:
        #    print("back to old splitter", node, var)
        #    self.nodes[node].splitter = old_splitter
        #    for n in self.nodes:
        #        try:
        #            self.nodes[n].W_update[:]=0
        #            self.nodes[n].h_update=0
        #        except AttributeError:
        #            pass
        #    return -1
        #self.nodes[node].splitter = old_splitter
        for n in self.nodes:
            try:
                self.nodes[n].W_update[:]=0
                self.nodes[n].h_update=0
            except AttributeError:
                pass
            
    #def lowest_level_spatial_updates(self, data, n_update_points=0, prob_map=(), split_var=(), both=True):
        
    def one_step_cyclic(self, data, scaling_factor=2, n_update_points=0, lowest=False, prob_map=(),
                       split_var=(), both=True, only_ends=False, levels=()):
        nodes = list(self.nodes.keys())
        depth = self.get_depth()
        if lowest:
            start=depth-1
        else:
            start=0
            
        if len(levels) == 0:
            levels = np.arange(depth-1, -1, -1)
        else:
            levels = np.sort(levels)[::-1]
            
        if len(split_var)>0:
            sdata = split_var
        else:
            sdata = data
        if n_update_points > 0:
            if len(prob_map)==0:
                self.simple_predict(sdata)
                prob_map = {}
                for n in nodes:
                    if len(n) >= 0:
                        prob_map[n] = self.nodes[n].map
                        #prob_map[n] = self.nodes[n].map>0
        #for level in range(depth-1, start-1, -1):
        for level in levels:
            for n in nodes:
                if len(n)==level:
                    if n+'1' in nodes:
                        self.nodes[n].h_update = 0 
                        self.nodes[n].W_update = np.zeros(sdata.shape[-1],
                                              dtype=np.float32)
                        #plt.plot(self.nodes[n].splitter[0])
                        if n_update_points > 0:
                            pm = prob_map[n] > 0
                            pm_max = np.minimum(prob_map[n+'0'].max(), prob_map[n+'1'].max())
                            #print(pm_max, 'pmmax', n)
                            #if pm_max > 0.1:
                                #update_pix = rand_sel(pm, n_update_points)
                            
                            update_pix = np.random.choice(np.sum(pm),
                                                          np.minimum(np.sum(pm), n_update_points))
                            self.subsamp = update_pix
                            if both:
                                self.node_grad(n, data[pm][update_pix], scaling_factor=scaling_factor, var='W',
                                              split_var=split_var,
                                              only_ends=only_ends)
                            self.node_grad(n, data[pm][update_pix], scaling_factor=scaling_factor, var='h',
                                          metropolis=False, split_var=sdata[pm][update_pix],
                                              only_ends=only_ends)
                            #else:
                            #self.nodes[n].splitter = [self.nodes[n].splitter[0]*0,
                            #                              self.nodes[n].splitter[1]*0]
                        else:
                            self.subsamp = []
                            if both:
                                self.node_grad(n, data, scaling_factor=scaling_factor, var='W',
                                              split_var=split_var,
                                              only_ends=only_ends)
                            self.node_grad(n, data, scaling_factor=scaling_factor, var='h',
                                          split_var=split_var,
                                              only_ends=only_ends)
                        #plt.plot(self.nodes[n].splitter[0])
                        #plt.show()
                       
    def node_stats(self):
        lengths ={}
        for n in self.nodes:
            try:
                lengths[len(n)] += 1
            except KeyError:
                lengths[len(n)] = 1
        for l in lengths:
            print(l, lengths[l], lengths[l]/2**l)
        return lengths

    def error_stats(self, data):
        depth = self.get_depth()
        errors = {}
        self.simple_predict(data)
        for l in range(depth+1):
            eL = self.remainder_at_level( data, l )
            total_err = np.sum(self.weights * eL.T**2)
            pointwise_err = total_err / len(data)
            errors[l] = pointwise_err
            print(l, errors[l])
        #for l in errors:
            
        return errors
    
    def excess_residual(self, node, data):
        resid = self.remainder_at_level(data, len(node))
        ldata = self.get_ldata(data)
        expected_errs = np.sum((ldata - ldata[self.neighbors])**2, axis=-1)
        #valid = expected_errs > 0
        resid_mag = np.sum(resid**2, axis=-1)
        excess_resid = np.sqrt(resid_mag/expected_errs)
        er_sum = np.sum(self.nodes[node].map[valid]*excess_resid)
        
        return er_sum
    
    def internal_split(self, node, data, split_var):
        resid = self.remainder_at_level(data, len(node))
        #ldata = self.get_ldata(data)
        #expected_errs = np.sqrt(np.sum((ldata - ldata[self.neighbors])**2, axis=-1))
        #print(expected_errs.min())
        #plt.imshow(np.sum(errs**2, axis=-1).reshape(100,100))
        #plt.show()
        #if self._use_norm:
        #    resid = (resid.T*np.sqrt(np.sqrt(np.sum(data**2, axis=-1)))).T
        U, s, Vh = ema.randomized_svd((resid.T*np.sqrt(self.nodes[node].map)*np.sqrt(self.weights)).T,
                                      n_components=1, random_state=0)
        #Un, sn, Vhn = ema.randomized_svd((errs.T*np.sqrt(self.nodes[node].map)*np.sqrt(self.weights)).T,
        #                              n_components=10, random_state=0)
        print(node, s)
        #print(self.internal_variance(node, data).mean())
        return s[0]
    
    def internal_pca(self, node, data):
        resid = self.remainder_at_level(data, len(node))
        #ldata = self.get_ldata(data)
        #expected_errs = np.sqrt(np.sum((ldata - ldata[self.neighbors])**2, axis=-1))
        #print(expected_errs.min())
        #plt.imshow(np.sum(errs**2, axis=-1).reshape(100,100))
        #plt.show()
        #if self._use_norm:
        #    resid = (resid.T*np.sqrt(np.sqrt(np.sum(data**2, axis=-1)))).T
        rel = self.nodes[node].map > 0
        #ldata = self.get_ldata(data[rel])
        U, s, Vh = ema.randomized_svd((resid[rel].T*np.sqrt(self.nodes[node].map[rel])*np.sqrt(self.full_weights[rel])).T,
                                      n_components=1, random_state=0)
        #Un, sn, Vhn = ema.randomized_svd((errs.T*np.sqrt(self.nodes[node].map)*np.sqrt(self.weights)).T,
        #                              n_components=10, random_state=0)
        print(node, s)
        #print(self.internal_variance(node, data).mean())
        return s[0]#-sn[0]#/np.sqrt(np.mean(s[-4:-1]))
   
    def internal_variance(self, node, data):
        return self.cross_variance(node, node, data)
    
    def cross_variance(self, node0, node1, data):
        #simple predict should have already run
        if len(node0)!=len(node1):
            print("need to choose nodes on the same level")
            return -1
        resid = self.remainder_at_level(data, len(node0))
        #if self._use_norm:
        #    resid = (resid*np.sqrt(np.sum(data, axis=-1)))
        
        return np.sum((self.weights*resid.T**2), axis=0)*self.nodes[node0].map*self.nodes[node1].map
                                        
    def rescale_all_nodes(self, split_level, split_var, less_than=True):
        nodes_2_rescale = [n for n in self.nodes if (n+'1') in self.nodes]
        depth = self.get_depth()
        for d in range(depth):
            for n in nodes_2_rescale:
                if len(n)==d:
                    self.rescale_node(split_level=split_level,
                                      node=n, 
                                      split_var=split_var,
                                      less_than=less_than)
                    
    def rescale_node(self, split_level, node, split_var, less_than=True):    
        """
        Assumes simple_predict has already been run
        needs to have "splitter", e.g. subnodes
        """
        s = self.nodes[node].splitter
        out = (s[0]@split_var.T + s[1]+1)/2
        sort_lab = np.argsort(np.abs(out-0.5))
        
        norm_sort = np.cumsum(self.full_weights[sort_lab]*self.nodes[node].map[sort_lab], dtype=np.float32)
        norm_sort /= (self.full_weights[sort_lab]*self.nodes[node].map.astype(np.float32)).sum()
        
        norm_cri = norm_sort > split_level
        try:
            scale_val = np.abs((out-0.5))[sort_lab][norm_cri].min()
        except ValueError:
            scale_val = 0.5
        if less_than:
            if scale_val > 0.5:
                return False
        else:
            if scale_val < 0.5:
                return False
            
        
        self.nodes[node].splitter = [s[0]*0.5/scale_val, 
                                     s[1]*0.5/scale_val]
        
        lmda = classify_from_partition(split_var, 
                                       self.nodes[node].splitter[0],
                                       self.nodes[node].splitter[1])
        self.nodes[node+'1'].lmda = lmda.astype(np.float16)
        self.nodes[node+'0'].lmda = (1-lmda).astype(np.float16)
        self.nodes[node+'1'].map = self.nodes[node].map*self.nodes[node+'1'].lmda
        self.nodes[node+'0'].map = self.nodes[node].map*self.nodes[node+'0'].lmda
        return True
    
    def blur_all_nodes(self, split_var):
        nodes_2_rescale = [n for n in self.nodes if (n+'1') in self.nodes]
        depth = self.get_depth()
        for d in range(depth):
            for n in nodes_2_rescale:
                if len(n)==d:
                    self.center_and_blur_node(node=n, 
                                      split_var=split_var)
    
    def center_and_blur_node(self, node, split_var, is_one=False):
        s = self.nodes[node].splitter
        m = self.nodes[node].map
        out = (s[0]@split_var[self.full_weights>0].T + s[1]+1)/2
        if is_one:
            o_range = (out[m==1].max() - out[m==1].min())
        else:
            o_range = (out.max() - out.min())
        o_offset = -(((s[0]/(o_range)@split_var[self.full_weights>0].T+1))).min()
        
        self.nodes[node].splitter = [s[0]/o_range, 
                                     o_offset]
        
        lmda = classify_from_partition(split_var, 
                                       self.nodes[node].splitter[0],
                                       self.nodes[node].splitter[1])
        self.nodes[node+'1'].lmda = lmda.astype(np.float16)
        self.nodes[node+'0'].lmda = (1-lmda).astype(np.float16)
        self.nodes[node+'1'].map = self.nodes[node].map*self.nodes[node+'1'].lmda
        self.nodes[node+'0'].map = self.nodes[node].map*self.nodes[node+'0'].lmda
        return True
    
    def node_clarity(self, node):
        clar = np.average(np.abs(self.nodes[node+'0'].lmda-0.5),
                   weights=self.nodes[node].map)*2
        return clar

    def mpmp(self):
        '''
        minimum percentage mixed pixels
        '''
        en=self.get_end_nodes()
        pmps = []
        for n in en:
            pmps.append(percentage_mixed_pixels(self.nodes[n].map))
        return np.min(pmps)
    
    def fix_end_classifiers(self, data, split_var, tol=1e-2, s_type='classifier'):
        end_nodes = self.get_end_nodes()
        

        other_nodes = [n for n in self.nodes if n not in end_nodes]
        print(end_nodes, other_nodes)
        classifiers = np.array([self.nodes[n].classifier for n in end_nodes])
        #print(classifiers)
        if s_type=='classifier':
            class_id = [np.argmin(np.sum((data-classifiers[i]).astype(np.float64)**2, axis=1)) for i in range(len(end_nodes))]
        elif s_type=='max':
            self.simple_predict(split_var)
            for n in end_nodes:
                i = 1
                sp = [0,0]
                while sp[1] == 0:
                    try:
                        sp = self.nodes[n[:-i]].splitter
                    except AttributeError:
                        i += 1
                self.nodes[n].lmda_exp = self.nodes[n[:-i]].map*((data@sp[0]+sp[1]+1)/2)
                if n[-i]=='0':
                    self.nodes[n].lmda_exp *= -1
            class_id = [np.argmax(self.nodes[n].lmda_exp) for n in end_nodes]
        print(class_id)
        self.subsamp = class_id
        self.simple_predict(split_var[class_id])
        s = np.array([self.nodes[n].map for n in end_nodes])
        prod = np.prod(np.array([s[i,i] for i in range(len(s))]))
        print(s, prod)
        while (1-prod)>tol:
            for i, cid in enumerate(class_id):
                self.subsamp = [cid]
                self.simple_predict(split_var[[cid]])
                rel_nodes = [n for n in other_nodes if (n==end_nodes[i][:len(n)])]
                for r in rel_nodes:
                    descendant = end_nodes[i][:(len(r)+1)]
                    sign = np.sign(int(descendant[-1]) - 0.5)
                    err = 1 - self.nodes[descendant].lmda[0] 
                    if err > 0:
                        #print("in0")
                        dot = np.dot(split_var[cid], split_var[cid])
                        delta = 2 * (err) * split_var[cid] / dot
                        s0 = self.nodes[r].splitter[0] + np.sign(sign) * delta
                        self.nodes[r].splitter = [s0, self.nodes[r].splitter[1]]
            self.subsamp = class_id
            self.simple_predict(split_var[class_id])
            s = np.array([self.nodes[n].map for n in end_nodes])
            prod = np.prod(np.array([s[i,i] for i in range(len(s))]))
        print(s)    
        
    def variance_minimizers(self, node, data, split_var=(), depth=0, aa=False, uncon=True):
        if depth==0:
            depth = len(node)
        pca = de.PCA(n_components=2)
        if len(split_var)>0:
            self.simple_predict(split_var)
        else:    
            self.simple_predict(data)
        #ldata = self.get_ldata(data)
        #expected_errs = np.sqrt(np.sum((ldata - ldata[self.neighbors])**2, axis=-1))
        self.get_full_weights()
        eL = self.remainder_at_level(data, depth)
        internal_residual = (eL.T*np.sqrt(self.nodes[node].map)*np.sqrt(self.weights)).T
        pca.fit(internal_residual)
        
        pca_eval = internal_residual @ pca.components_.T[:,0]
        plt.imshow(pca_eval.reshape(self.plot_size), cmap='bwr')
        plt.colorbar()
        plt.show()
        
        pca_eval = internal_residual @ pca.components_.T[:,1]
        plt.imshow(pca_eval.reshape(self.plot_size), cmap='bwr')
        plt.colorbar()
        plt.show()
        
        pca_eval = (internal_residual @ pca.components_.T)
        #pca_eval += pca_eval.min()
        
        #print('md', data[:10,0], split_var[:10,0])
        rsplit = self.remainder_splitter(pca_eval, split_var, tol=(1e-2, 1e-2), A_tol=0.1,
                                        nmap=self.nodes[node].map)
        
        rsplit.simple_predict(split_var)
        e = rsplit.remainder_at_level(pca_eval, 1)
        d_rem = np.sum(pca_eval**2 - e**2)
        #plt.plot(rsplit.nodes[''].splitter[0])
        #plt.show()
        rsplit.improvement = d_rem
        
        return rsplit
                               
        
                               
        pca_eval -= np.average(pca_eval)
        pca_plus = pca_eval > 0
        
        #if self._use_norm:
        #    ldata = (data.T/np.sqrt(np.sum(data**2, axis=1))).T
        #else:
        #    ldata = data
        
        #plt.imshow(self.nodes[node].map.reshape(self.plot_size))
        #plt.colorbar()
        #plt.show()
        denom = np.sum(self.nodes[node].map[pca_plus].astype(np.float64))
        print('denom', denom, self.nodes[node].map.max(), self.nodes[node].map.dtype)
        top_avg = np.mean(pca_eval[pca_plus].T)#*self.nodes[node].map[pca_plus])/denom
        
        #top_avg = np.sum(ldata[pca_plus].T*self.nodes[node].map[pca_plus]**2, axis=1)/denom
        
        denom = np.sum((self.nodes[node].map[~pca_plus]).astype(np.float64))
        bot_avg = np.mean(pca_eval[~pca_plus].T)#*self.nodes[node].map[~pca_plus])/denom

        #plt.plot(top_avg)
        #plt.plot(bot_avg)
        #plt.show()
        
        top_pix = pca_eval > top_avg
        bot_pix = pca_eval < bot_avg#np.argmin(np.sum((ldata-bot_avg)**2, axis=1))
        
        denom = np.sum(self.nodes[node].map[top_pix].astype(np.float64))
        top_avg2 = np.mean(pca_eval[top_pix].T)#*self.nodes[node].map[top_pix])/denom
        
        denom = np.sum((self.nodes[node].map[bot_pix]).astype(np.float64))
        bot_avg2 = np.mean(pca_eval[bot_pix].T)#*self.nodes[node].map[bot_pix])/denom

        top_pix2 = pca_eval > top_avg2
        bot_pix2 = pca_eval < bot_avg2
        
        test1 = svm.LinearSVC(max_iter=100000, dual=False, tol=1e-4)#,
                              #C = np.sqrt(1/np.sum(top_pix|bot_pix)))
        print(top_avg2, bot_avg2, np.sum(top_pix2), np.sum(bot_pix2))
        #Xs = data[top_pix | bot_pix]
        #ys = top_pix[top_pix | bot_pix]
        #test1.fit(Xs, ys)
        
        
        xmax, xmin = np.argmax(self.nodes[node].map*pca_eval),\
                     np.argmin(self.nodes[node].map*pca_eval)
        #xmax, xmin = np.argmax(pca_eval), np.argmin(pca_eval)
        #c, i  = test1.coef_[0], test1.intercept_
        #vals = data[top_pix2 | bot_pix2] @ c
        #cmin = np.min(vals)
        if len(split_var) > 0:
            Xs = data[top_pix2 | bot_pix2]
            ys = top_pix[top_pix2 | bot_pix2]
            test1.fit(Xs, ys)
            return test1.coef_[0], test1.intercept_#classifiers_2_svm(split_var[xmax], split_var[xmin])#test1.coef_[0], test1.intercept_#classifiers_2_svm(split_var[xmax], split_var[xmin])#est1.coef_[0], test1.intercept_#classifiers_2_svm(split_var[xmax], split_var[xmin])
        else:
            Xs = split_var[top_pix2 | bot_pix2]
            ys = top_pix[top_pix2 | bot_pix2]
            test1.fit(Xs, ys)
            return test1.coef_[0], test1.intercept_#classifiers_2_svm(data[xmax], data[xmin])#est1.coef_[0], test1.intercept_#classifiers_2_svm(data[xmax], data[xmin])#test1.coef_[0], test1.intercept_##
        #x_guess = test1.predict(Xs)
        #print(np.mean(x_guess[ys==1]==ys[ys==1]))
        #print(np.mean(x_guess[ys==0]==ys[ys==0]))
        #d_out = data@test1.coef_.T + test1.intercept_
        #d_out2 = classify_from_partition(data, test1.coef_,test1.intercept_)

        
        #return top_pix, bot_pix
        #int_var = self.internal_variance(node, data)
        #worst_point = np.argmax(int_var)
        #fd = lambda x: fractional_distance(data[worst_point],
        #                                   self.nodes[node].classifier, x)
        #fddat = fd(data)
        #sorted_fd = []
        #for i in np.argsort(fddat):
        #    sorted_fd.append([fddat[i], self.nodes[node].map[i]**2])
        #sorted_fd = np.array(sorted_fd)
        #cumulative_fd = np.array([np.sum(sorted_fd[:i,1]) for i in range(len(sorted_fd))])
                                        
        #targ = cumulative_fd.max()/4    
        #cut1 = np.argmin((sorted_fd[:,1]+1e-6)**-8*(np.abs(cumulative_fd-targ)+targ/4)**2)
        #targ = 3*cumulative_fd.max()/4
        #cut2 = np.argmin((sorted_fd[:,1]+1e-6)**-8*(np.abs(cumulative_fd-targ)+targ/4)**2)
        #return #np.argsort(fddat)[cut1], np.argsort(fddat)[cut2]
        
    def untangle_endmembers(self, image, s_type='max'):
        """
        note that this only works if there are linearly separable pixels within each endmembers cluster
        """
        self.simple_predict(image)
        self.binarize_lmdas()
        self.lmda_2_map()
        self.update_from_level_S_V(image, beta = 0, alg='simple', a_len_max=0.01, n_update_points=0,
                         attenuation = 1, levels=(-1,), occs=(), split_var=image)
        self.fix_end_classifiers(self.get_ldata(image), image, s_type=s_type)
        
    def both_sides_pure(self, node):
        #assume that it has the children
        minus = np.sum(self.nodes[node+'0'].map == 1) > 0
        plus = np.sum(self.nodes[node+'1'].map == 1) > 0
        if (plus and minus):
            print(node, " is pure")
        return (plus and minus)

    def sparse_grow_node(self, data, split_var, to_grow):
        self.grow_node(to_grow)
        a, e0, e1 = quick_split(data[self.nodes[to_grow].map==1])
        splitter_a = classifiers_2_svm(e0, e1)
        out = (data@splitter_a[0]+splitter_a[1]+1)/2
        mean0 = np.mean(split_var[out<0.5], axis=0)
        mean1 = np.mean(split_var[out>0.5], axis=0)
        splitter = classifiers_2_svm(mean0, mean1)
        self.nodes[to_grow].splitter = splitter
        
    def equiliberate(self, image, n_runs=1e-3, n_pts=1000, sampling_points=(), obj_record=[]):
        self.switch_training(image, beta=0, tol=1e-12, n_update_points=n_pts, 
                                 scaling_factor=2, sampling_points=sampling_points,
                                 alg='simple', obj_record=obj_record, A_tol=1e-4, max_iter=n_runs,
                                 A_protection=False)
        
    def all_classifiers(self):
        classifiers = []
        for n in self.nodes:
            classifiers.append(np.copy(self.nodes[n].classifier))
        return classifiers
    
    def get_scores(self, gt_map, gt_e, S, E, show=False):
        rows, cols = align_spectra(gt_map, S)
        L = len(gt_map)
        IoU = np.zeros(L)
        rmse_a = np.zeros(L)
        spec_ang = np.zeros(L)
        sparsity = np.zeros(L)
        diff = 0
        fig, ax = plt.subplots(3, len(gt_map))
        fig2, ax2 = plt.subplots(1)
        for j, x in enumerate(cols):
            i = rows[j]
            print(rows, cols)
            intersection = np.sum(np.minimum(S[i], gt_map[x]))
            union = np.sum(np.maximum(S[i], gt_map[x]))
            IoU[j] = intersection/union
            rmse_a[j] = np.sqrt(np.mean((S[i] - gt_map[x])**2))*100
            spec_ang[j] = np.sum(E[i]*gt_e[x])
            spec_ang[j] /= np.sqrt(np.sum(E[i]**2))
            spec_ang[j] /= np.sqrt(np.sum(gt_e[x]**2))
            spec_ang[j] = np.arccos(spec_ang[j])*(180/np.pi)
            diff += np.sum(np.abs(S[i] - gt_map[x]))/2
            ax[0,j].imshow(gt_map[x].reshape(self.plot_size))
            ax[1,j].imshow(S[i].reshape(self.plot_size))
            ax[2,j].imshow((S[i]-gt_map[x]).reshape(self.plot_size), vmin=-1, vmax=1, cmap='bwr')
            a = ax2.plot(E[i]/np.sqrt(np.sum(E[i]**2)))
            ax2.plot(gt_e[x]/np.sqrt(np.sum(gt_e[x]**2)), '--',color=a[0].get_color())
            for b in ax[:,j]:
                b.spines['bottom'].set_color(a[0].get_color())
                b.spines['top'].set_color(a[0].get_color()) 
                b.spines['right'].set_color(a[0].get_color())
                b.spines['left'].set_color(a[0].get_color())
                plt.setp(b.spines.values(), linewidth=2)


            n = len(S[i])
            sparsity[j] = (np.sqrt(n)-np.sum(S[i],axis=0)/np.sqrt(np.sum(S[i]**2,axis=0)))/(np.sqrt(n)-1)

        for a in ax:
            for b in a:
                b.set_xticks([])
                b.set_yticks([])
        fig.tight_layout()
        plt.show()

        scoreD = {
            'sa': spec_ang,
            'tot_d': diff,
            'rmse': rmse_a,
            'IoU': IoU,
            'sparsity': sparsity
        }
        
        return scoreD
    
    def get_trimmed_network(self, accepted_nodes):
        en = self.rel_end_nodes(accepted_nodes)
        newdeh = copy.deepcopy(self)
        L = max([len(n) for n in newdeh.nodes])
        del newdeh.nodes
        newdeh.nodes = {n:copy.deepcopy(self.nodes[n]) for n in accepted_nodes}
        for n in en:
            print(n)
            newdeh.nodes[n]=copy.deepcopy(self.nodes[n])
            del newdeh.nodes[n].splitter
            l = len(n)
            for i in range(L-l):
                newdeh.nodes[n+i*'0'] = copy.deepcopy(self.nodes[n])
        return newdeh
    
    def en_classifier_diffs(self, node):
        en = self.get_end_nodes()
        classi_diffs = {}
        for n in en:
            if n!=node:
                classi_diffs[n] = np.sum((self.nodes[n].classifier-self.nodes[node].classifier)**2)
        return classi_diffs

    def en_min_max(self):
        mins = {}
        for n in self.get_end_nodes():
            #print(en_classifier_diffs(deh2, n))
            en_diffs = self.en_classifier_diffs(n)
            mins[n] = en_diffs[min(en_diffs, key=en_diffs.get)]
        return (mins[min(mins, key=mins.get)], mins[max(mins, key=mins.get)])
    
    def sparse_grow_node(self, data, split_var, to_grow):
        self.grow_node(to_grow)
        a, e0, e1 = quick_split(data[self.nodes[to_grow].map==1])
        splitter_a = classifiers_2_svm(e0, e1)
        out = (data@splitter_a[0]+splitter_a[1]+1)/2
        mean0 = np.mean(split_var[out<0.5], axis=0)
        mean1 = np.mean(split_var[out>0.5], axis=0)
        splitter = classifiers_2_svm(mean0, mean1)
        self.nodes[to_grow].splitter = splitter
        
    def equiliberate(self, image, n_runs=100, n_pts=1000, sampling_points=(), obj_record=[],
                     scaling_factor=2, epsilon = 0.1):
        i = 0
        for i in range(n_runs):
            if self.verbose:
                print("equiliberate run: ", i, "of ", n_runs)
            self.rescale_all_nodes(epsilon, image, less_than=False)
            self.switch_training(image, beta=0, tol=1e-12, n_update_points=n_pts, 
                                     scaling_factor=scaling_factor, sampling_points=sampling_points,
                                     alg='simple', obj_record=obj_record, A_tol=1e-4, max_iter=1,
                                     A_protection=False, only_ends=False)
            
    def rel_end_nodes(self, nodes_to_split):
        ren = []
        for node in nodes_to_split:
            if (node + '1' in self.nodes) & (node + '1' not in nodes_to_split):
                ren.append(node+'1')
            if (node + '0' in self.nodes) & (node + '0' not in nodes_to_split):
                ren.append(node+'0')
        return ren
    
    def node_scoresI(self, data, nodes_to_split):
        ren = self.rel_end_nodes(nodes_to_split)
        scores = {}
        #print(ren)
        for n in ren:
            n2s2 = copy.deepcopy(nodes_to_split)
            n2s2.append(n)
            ren2 = self.rel_end_nodes(n2s2)
            #print(ren2, n2s2)
            err = copy.deepcopy(self.get_ldata(data))
            for m in ren2:
                err -= np.outer(self.nodes[m].map,self.nodes[m].classifier)
            err = (self.full_weights*err.T).T
            scores[n] = np.mean(np.sum(err**2, axis=1))
        print(scores)
        return scores
    
    def adia_split_node(self, data, accepted_nodes):
        scores = self.node_scoresI(data, accepted_nodes)
        to_accept = min(scores, key=scores.get)

        d_norm = (data.T/np.sqrt(np.sum(data**2, axis=-1))).T
        self.sparse_grow_node(d_norm, data, to_accept+'0')
        self.sparse_grow_node(d_norm, data, to_accept+'1')
        return to_accept
    
    def sparsify(self, data, sampling_points, obj_record, n_points, mpp_tol=0.05, step_delta = 0.01,
             reg_max=1):
        self.reg = 0
        self.use_bsp = True
        scaling_factor = 1
        #d_scale = diff_scale(DEH)
        #rel_node = min(d_scale, key=d_scale.get)
        #rel_scale = d_scale[rel_node]/(100*len(data))
        #DEH.set_mu(0)
        #DEH.delta_mu = rel_scale
        #DEH.increment_mu()
        self.sparsity_sweep(data, sampling_points, obj_record, n_points, step_delta, reg_max)
        deh_mpp = self.mpp(data)
        while deh_mpp > mpp_tol:
            if self.verbose:
                print("sparsifying loop")
                print("deh_mpp:", deh_mpp, " > mmp_tol: ", mpp_tol)
            scaling_factor = np.maximum(0, scaling_factor - mpp_tol)
            scale = self.max_node_scale(data)
            scale *= scaling_factor
            self.rescale_all_nodes(scale, data, less_than=True)
            for n in self.splitting_nodes():
                if not self.both_sides_pure(n):
                    self.rescale_node(mpp_tol, n, data, less_than=False)

            self.sparsity_sweep(data, sampling_points, obj_record, n_points, step_delta, reg_max )
            deh_mpp = self.mpp(data)
        self.use_bsp = False
    
    def mpp(self, image):
        S = self.simple_predict(image)
        return 1-(S==1).sum() / len(image)
    
    def sparsity_sweep(self, data, sampling_points, obj_record, n_update_points, step_delta=0.01, reg_max = 1):
        deh_mpp = 0.1
        self.reg = 0
        #DEH.sparsifying=True
        while self.reg < reg_max:
            self.reg += step_delta
            #DEH.only_ends=True
            self.switch_training(data, beta=0, tol=1e-12, n_update_points=n_update_points, 
                                         scaling_factor=2, sampling_points=sampling_points,
                                         alg='simple', obj_record=obj_record, A_tol=1e-4, max_iter=1,
                                         A_protection=False, only_ends=False)
            #DEH.only_end=False
            deh_mpp = self.mpp(data)
            if self.verbose:
                print(self.reg, deh_mpp)
            #reg_mix.append([DEH.reg, deh_mpp])
            #self.display_level(self.get_depth())
        #DEH.sparsifying=False
        self.reg = 0
        
    def pick_to_split(self, data, accepted_nodes):
        scores = self.node_scoresI(data, accepted_nodes)
        to_accept = min(scores, key=scores.get)
        return to_accept
    
    def do_split(self, data, split_node):
        d_norm = (data.T/np.sqrt(np.sum(data**2, axis=-1))).T
        self.sparse_grow_node(d_norm, data, split_node+'0')
        self.sparse_grow_node(d_norm, data, split_node+'1')
        
    def adia_add_nodeII(self, data, accepted_nodes, sampling_points=(), obj_record=(), n_points=0, reg_max=0.2,
                 n_runs=20, scales=4, mpp_tol=0.05, save=False, save_name='default'):
        # Equiliberate 2x
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, n_runs=n_runs, epsilon=mpp_tol, scaling_factor=4, n_pts=n_points)
        self.reg = 0
        self.equiliberate(data, n_runs=n_runs, epsilon=mpp_tol, scaling_factor=2, n_pts=n_points)
        
        self.simple_predict(data)
        self.display_level(self.get_depth())
        # pick node to split
        to_split = self.pick_to_split(data, accepted_nodes)
        
        
        # sparsify
        self.use_bsp = True
        self.sparsify(data, sampling_points=sampling_points, obj_record=obj_record,
                 n_points=n_points, reg_max=reg_max, mpp_tol=mpp_tol)
        self.use_bsp = False
        
        #split node
        self.do_split(data, to_split)
        
        #equiliberate
        self.reg = 0
        self.equiliberate(data, n_runs=n_runs, epsilon=0, scaling_factor=4, n_pts=n_points)
        
        
        #sparsify again
        self.use_bsp = True
        self.sparsify(data, sampling_points=sampling_points, obj_record=obj_record,
                 n_points=n_points, reg_max=reg_max, mpp_tol=mpp_tol)
        self.use_bsp = False
        
        accepted_nodes.append(to_split)
        return accepted_nodes
        
    def adia_add_node(self, data, accepted_nodes, sampling_points=(), obj_record=(), n_points=0, reg_max=0.2,
                 n_runs=20, scales=4, mpp_tol=0.05, save=False, save_name='default'):
        #sparsify network
        self.use_bsp = True
        self.sparsify(data, sampling_points=sampling_points, obj_record=obj_record,
                 n_points=n_points, reg_max=reg_max, mpp_tol=mpp_tol)
        self.save(save_name+'_' + 'sp' + '_' + str(len(self.get_end_nodes()))+'.h5')
        #split node
        self.simple_predict(data)
        #self.binarize_lmdas()
        #self.lmda_2_map()
        accepted = self.adia_split_node(data, accepted_nodes)
        #equiliberate nodes
        self.reg = 0
        self.set_mu(0)
        self.use_bsp=False
        print("first equilibration")
        self.equiliberate(data, n_runs=n_runs, epsilon=mpp_tol, scaling_factor=4)
        self.rescale_all_nodes(mpp_tol, data, less_than=False)
        print("second equilibration")
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, n_runs=n_runs, epsilon=mpp_tol, scaling_factor=2)

        #de-sparsify nodes

        #blur_network(self, data, reg_max, mpp_tol, scales, n_update_points=n_points, n_runs=n_runs)
        #de-sparsed equiliberation
        self.reg = 0
        print("third equiliberation")
        self.equiliberate(data, n_runs=n_runs, epsilon=0.0, scaling_factor=2 )
        accepted_nodes.append(accepted)
        return accepted_nodes

    def adia_grow_network(self, data, n_nodes, n_update_pts= (0,), mpp_tol=0.05, saturation=(), sampling_points=(),
                      reg_max=0.2, n_runs=20, scales=5, save=False, save_name='default', step_size=0.001):
        obj_record=[[0]]
        self.training='adiabatic'
        self.start=time.time()
        self.use_norm(True)
        self.n_update_pts = n_update_pts[0]
        self.training='grow_network_single'
        self.neighbors = quick_nn(data.reshape(self.plot_size + (-1,)), k_size=1).flatten()
        self.set_neighbor_weights(data)
        if len(saturation)==len(data):
            self.full_weights[saturation] = 0
        self.get_full_weights()
        self.parameter_initialization(data)


        d_norm = (data.T/np.sqrt(np.sum(data**2, axis=-1))).T
        self.sparse_grow_node(d_norm, data, '')
        accepted_nodes = ['']
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, n_pts=n_update_pts[0], epsilon=mpp_tol,
                     scaling_factor = 1)
        self.use_bsp = True
        self.sparsify(data, sampling_points=sampling_points, obj_record=obj_record,
                 n_points=n_update_pts[0], reg_max=reg_max, mpp_tol=mpp_tol)
        self.use_bsp=False
        self.reg=0
        for n in accepted_nodes:
            if n+'11' in self.nodes:
                pass
            else:
                self.sparse_grow_node(d_norm, data, n+'1')
            if n+'01' in self.nodes:
                pass
            else:
                self.sparse_grow_node(d_norm, data, n+'0')
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, n_pts=n_update_pts[0], epsilon=mpp_tol,
                     scaling_factor = 2)
        self.rescale_all_nodes(mpp_tol, data, less_than=False)
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, n_pts=n_update_pts[0], epsilon=mpp_tol,
                     scaling_factor = 2)


        self.save(save_name+'_' + 'eq' + '_' + str(len(self.get_end_nodes()))+'.h5')
        self.simple_predict(data)
        self.display_level(2)
        while len(self.rel_end_nodes(accepted_nodes)) < n_nodes:
            accepted_nodes = self.adia_add_nodeII(data, accepted_nodes,
                          n_points=n_update_pts[0], mpp_tol=mpp_tol, reg_max=reg_max,
                          n_runs=n_runs, scales=scales, obj_record=obj_record, save=save,
                          save_name=save_name)
            self.save(save_name+'_' +'eq' + '_' + str(len(self.get_end_nodes()))+'.h5')

        return obj_record, accepted_nodes
    
    def accepted_network_stablization(self, data, n_runs=100, n_pts=(0,), obj_record=(), sampling_points=(),
                                 mpp_tol=0.05, step_delta=0.01, reg_max=0.2, name='default'):
        if len(obj_record)==0:
            obj_record = [0]

        if len(sampling_points)==0:
            sampling_points = np.arange(len(data))

        self.only_ends=False
        self.aa = False

        #print("sf1-4e025")
        #self.reg = -self.en_min_max()[0]/4
        #self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =0.25,
        #        n_pts=n_pts[0], sampling_points=sampling_points)

        #counter-bias with scaling factor of 2 and -/4 bias and epsilon = 0.25
        print("\n----sf2-4e025----\n")
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=2, epsilon =0.25,
                n_pts=n_pts[0], sampling_points=sampling_points)

        self.simple_predict(data)
        # self.display_level(self.get_depth())
        #counter-bias with scaling factor of 4 and -/4 bias and epsilon = 0.5
        print("\n----sf4-4e05----\n")
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=4, epsilon =0.5,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.simple_predict(data)
        # self.display_level(self.get_depth())


        #counter-bias with scaling factor of 4 and -/4 bias and epsilon = 0.25
        print("\n----sf4-4e025----\n")
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=4, epsilon =0.25,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.simple_predict(data)
        # self.display_level(self.get_depth())


        #counter-bias with scaling factor of 4 and -/4 bias and epsilon = 0
        print("\n----sf4-4e0----\n")
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=4, epsilon =0.0,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.simple_predict(data)
        # self.display_level(self.get_depth())


        #counter-bias with only end and -/4 bias
        print("\n----oe-4----\n")
        self.only_ends = True
        self.reg = -self.en_min_max()[0]/4
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =0.0,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.only_ends = False
        self.simple_predict(data)
        # self.display_level(self.get_depth())


        #only ends and no counter-bias
        print("\n----oe----\n")
        self.reg = 0
        self.only_ends = True
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =0.0,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.simple_predict(data)
        # self.display_level(self.get_depth())

        print("sparsify")
        self.sparsify(data, sampling_points=sampling_points, obj_record=obj_record,
                 n_points=n_pts[0], reg_max=reg_max, mpp_tol=mpp_tol, step_delta=step_delta)
        
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =0.1,
                n_pts=n_pts[0], sampling_points=sampling_points)
        
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =0,
                n_pts=n_pts[0], sampling_points=sampling_points)
        #only ends with +/4 bias
        ## sparsify again
        #sparsify(DEH, data, sampling_points=sampling_points, obj_record=obj_record, n_points=n_pts[0],
        #        mpp_tol=mpp_tol, step_delta = step_delta, reg_max=reg_max)
        #self.reg = self.en_min_max()[0]/4
        #self.only_ends = True
        #self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
        #        n_pts=n_pts[0], sampling_points=sampling_points)
        #self.simple_predict(data)
        #self.display_level(self.get_depth())


        #self.reg = self.en_min_max()[0]/2
        #self.only_ends = True
        #self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
        #        n_pts=n_pts[0], sampling_points=sampling_points)
        #self.simple_predict(data)
        #self.display_level(self.get_depth())
        
        #only ends and no counter bias
        DEH2 = copy.deepcopy(self)
        DEH2.only_ends = True
        DEH2.rescale_all_nodes(mpp_tol, data, less_than=False)
        DEH2.reg = 0
        DEH2.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
                n_pts=n_pts[0], sampling_points=sampling_points)
        DEH2.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
                n_pts=n_pts[-1], sampling_points=sampling_points)
        DEH2.simple_predict(data)
        # DEH2.display_level(self.get_depth())

        #save ppa version
        DEH2.training = 'PPA'
        ppa_name = name+'_' + 'ppa' +'.h5' 
        print("\n----Save ppa----\n")
        DEH2.save(name+'_' + 'ppa' +'.h5')
        del DEH2

        #turn on aa
        self.aa = True
        self.only_ends = True
        self.reg = 0 
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
                n_pts=n_pts[0], sampling_points=sampling_points)
        self.equiliberate(data, obj_record=obj_record, n_runs=n_runs, scaling_factor=1, epsilon =mpp_tol,
                n_pts=n_pts[-1], sampling_points=sampling_points)
        self.simple_predict(data)
        # self.display_level(self.get_depth())

        #save aa version
        self.training = 'AA'
        aa_name =  name+'_' + 'aa'  +'.h5'
        print("\n----Save aa----\n")
        self.save(name+'_' + 'aa' +'.h5')

        return ppa_name, aa_name
    
    def get_node_scale(self, node, split_var, less_than=True):    
        """
        Assumes simple_predict has already been run
        needs to have "splitter", e.g. subnodes
        """
        s = self.nodes[node].splitter
        out = (s[0]@split_var.T + s[1]+1)/2
        sort_lab = np.argsort(np.abs(out-0.5))
        all_pix = (self.full_weights*self.nodes[node].map.astype(np.float32)).sum()
        relevant = np.abs(out-0.5)<0.5
        rel_pix = np.sum(self.full_weights[relevant]*self.nodes[node].map[relevant].astype(np.float32))

        #norm_sort = np.cumsum(DEH.full_weights[sort_lab]*DEH.nodes[node].map[sort_lab], dtype=np.float32)
        #norm_sort /= (DEH.full_weights[sort_lab]*DEH.nodes[node].map.astype(np.float32)).sum()

        #norm_cri = norm_sort > split_level
        #scale_val = np.abs((out-0.5))[sort_lab][norm_cri].min()

        return rel_pix/all_pix

    def max_node_scale(self, split_var):
        max_scale = 0
        for n in self.nodes:
            try:
                scale = self.get_node_scale(n, split_var)
            except AttributeError:
                scale = 0
            max_scale = np.maximum(max_scale, scale)
        return max_scale
    
    def splitting_nodes(self):
        #simple predict must be run first
        p_nodes = []
        for n in self.nodes:
            try:
                if (np.sum(self.nodes[n+'1'].map)>0)&(np.sum(self.nodes[n+'0'].map)>0):
                    p_nodes.append(n)
            except KeyError:
                pass
        return p_nodes
    
def quick_split(data, tol=1e-6, weights = (), ppa=True):
    if len(weights)==0:
        j0 = data.mean(axis=0)
        print(j0.shape)
        j0 = data[np.argsort(np.sum((data-j0)**2, axis=1))[len(data)//2]]
        o_err1 = np.sum((data)**2)
        o_err = np.sum((data - j0)**2)
    else:
        j0 = np.average(data, axis=0, weights=weights)
        o_err1 = np.sum(weights*((data)**2).T)
        o_err = np.sum(weights*((data - j0)**2).T)
    j1 = data[np.argsort(np.sum((data-j0)**2, axis=1))[len(data)//2]]

    
    delta = 1
    
    n_err = o_err
    o_err = o_err1
    delta = (o_err-n_err)/(o_err+n_err)
    first_flag = True
    while delta > tol:
        out = svm_from_classifiers(data.T,j0, j1)
        if first_flag:
            out_sort = np.sort(out)
            boundary = out_sort[len(data)//2]
        else:
            boundary = 0.5
        if len(weights)==0:
            j0 = np.mean(data[out<boundary], axis=0)
            j1 = np.mean(data[out>=boundary], axis=0)
        else:
            j0 = np.average(data[out<boundary], axis=0, weights=weights[out<0.5])
            j1 = np.average(data[out>=boundary], axis=0, weights=weights[out>=0.5])
        if ppa:
            j0 = data[np.argsort(np.sum((data-j0)**2, axis=1))[0]]
            j1 = data[np.argsort(np.sum((data-j1)**2, axis=1))[0]]
        recon = np.outer(out>boundary,j1) + np.outer(out<=boundary,j0)
        o_err = n_err
        if len(weights)==0:
            n_err = np.sum((data - recon)**2)
        else:
            n_err = np.sum(weights*((data - recon)**2).T)
        delta = (o_err-n_err)/(o_err+n_err)
        first_flag = False
    #print(n_err/o_err1, delta, n_err, o_err)
    
    
    return o_err1-n_err, j0, j1            
            
def fractional_distance(far, close, q):
    return np.dot(far-q, far-close)/np.sum((far-close)**2)
    
def find_opt_beta(intercepts, initial_include, num_weights, denom_weights, target = 0, title=""):
    '''
    Approximate the beta for which the slope vanishes
    '''
    EPS = 1e-16
    
    old_beta = 0
    num = np.sum(num_weights[initial_include])
    denom = np.sum(denom_weights[initial_include])
    #contrib = np.zeros_like(num_weights)
    #contrib[initial_include] = num_weights[initial_include]
    args = np.argsort(intercepts)
    slope = num
    initial_slope = slope
    #print(slope)
    i = 0
    shift = 0
    beta = 0
    old_slope = 0
    old_beta = 0
    min_int = np.sum(initial_include)
    #print("initial beta guess", -num/denom)
    betas = [old_beta]
    #new_slopes = [np.sum(contrib)]
    slopes = [slope]
    included = [min_int]
    init_slope_guess = slope
    baseline = [slope]
    #print("intercepts", np.sort(intercepts)[:10])
    while (slope < (-initial_slope*target))&(i<len(args)):
        older_beta = old_beta
        old_beta = beta
        old_slope = slope
        ID = args[i]
        beta = intercepts[ID]
        deltabeta = beta - old_beta
        shift += deltabeta*denom
        num_weights[initial_include] += denom_weights[initial_include]*deltabeta
        #contrib[initial_include] += deltabeta*denom_weights[initial_include]
        initial_include[ID] = ~initial_include[ID]
        min_int = np.minimum(min_int, np.sum(initial_include))
        included.append(np.sum(initial_include))
        if min_int < 1:
            slope = 0
            beta = old_beta
        else:
            if initial_include[ID]:
                #contrib[ID] += num_weights[ID]
                new_num = num + num_weights[ID]
                denom_new = denom + denom_weights[ID]
            else:
                #contrib[ID] = 0
                new_num = num - num_weights[ID]
                denom_new = denom - denom_weights[ID]
            num = new_num
            denom= denom_new
            slope = new_num + shift
        i += 1
        if i>=(len(args)-1):
            slope=0
        #new_slopes.append(np.sum(contrib))
        betas.append(beta)
        slopes.append(slope)
        baseline.append(new_num)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(betas, slopes, '.-', color='red')
    #ax.plot(betas, baseline, '.-', color='blue')
    #ax.plot(betas, new_slopes, '.-', color='pink')
    #ax.set_title(title)
    #plt.axhline(y=0, color='grey', linestyle='-')
    #ax2 = ax.twinx()
    #ax2.plot(betas, included,'.-', color='black', label='mixedpix')
    #plt.legend()
    #plt.show()
    #print("min ints", min_int) 
    return np.minimum((beta+old_beta)/2,np.maximum(old_beta-old_slope/denom, 0))
                                     #np.minimum(np.maximum(beta-EPS,0)#, np.maximum(old_beta-old_slope/denom,0))#(old_beta + beta) / 2

def classify_from_partition(data, coef, intercept, threshold=0, spread=1):
    base = (coef@data.T + intercept + 1)/(2)
    trimmed = np.array(np.maximum(np.minimum(base, 1.0),0), dtype=np.float32)
    return trimmed
    
def svm_from_classifiers(image, classifier0, classifier1):
    #print(image.shape, classifier0.shape)
    pw = (classifier1 - classifier0)
    w = 2 * pw / np.dot(pw, pw)
    d = -np.dot(pw, classifier1 + classifier0)/ np.dot(pw, pw)
    
    #print(image.shape, w.shape)
    lambdas = w@image + d
    lambdas = (lambdas + 1)/2
    lambdas = np.minimum(1, lambdas)
    lambdas = np.maximum(0, lambdas)
    return lambdas

def sparsity(S):
    n = len(S)
    sparsity = (np.sqrt(n)-np.sum(S,axis=0)/np.sqrt(np.sum(S**2,axis=0)))/(np.sqrt(n)-1)
    return sparsity.mean()

def classifiers_2_svm(classifier0, classifier1):
    pw = (classifier1 - classifier0)
    
    w = 2 * pw / np.dot(pw, pw)
    if np.isnan(w).any():
        print("theres a nan ", pw, classifier0, classifier1)
    d = -np.dot(pw, classifier1 + classifier0)/ np.dot(pw, pw)
    
    return w, d

def scaled_2class_svm(resid_image, scaling_factors, classifier0, classifier1):
    scaled_image = (resid_image).T / scaling_factors
    scaled_lambdas = svm_from_classifiers(scaled_image, classifier0, classifier1)
    two_classes = [1-scaled_lambdas, scaled_lambdas]
    two_classes *= scaling_factors
    return two_classes

def w_avg(pix, weights):
    num = np.sum(pix.T*weights, axis=1)
    denom = np.sum(weights)
    return num/denom
    
def group_err(pix, weights):
    avg = w_avg(pix, weights)
    diff = pix - avg
    err = weights*np.sum(diff**2, axis=1)
    return np.sum(err)

def regroup_1it(pix, weights, groups):
    #gen f-matrix
    f_matr = np.zeros((len(groups),len(groups)))
    for i in range(len(groups)):
        for j in range(i,len(groups)):
            pts = flat(groups[i][0])+flat(groups[j][0])
            #print(pts)
            f_matr[i,j] = group_err(pix[pts], weights[pts]) 
            f_matr[i,j] -= groups[i][1] + groups[j][1]
            f_matr[j,i] = f_matr[i,j] 
        f_matr[i,i] = np.inf
        
    #compute nearest-neighbors
    NN = np.zeros(f_matr.shape, dtype=bool)
    for j in range(len(f_matr)):
        k = np.argmin(f_matr[j])
        NN[j, k] = True
    RNN = NN&NN.T
    
    new_groups = [x for i, x in enumerate(groups) if RNN[i].sum() == 0]
    for i in range(len(RNN)):
        if RNN[i,:i].sum() == 1:
            k = np.argmax(RNN[i,:i])
            new_groups.append([[groups[i][0]] + [groups[k][0]],
                             f_matr[i,k] + groups[i][1] + groups[k][1]])
    
    return new_groups

def rand_sel(qprob_map, number_of_pixels):
    '''
    randomly selects number_of_pixels according to qprob_map probability distribution
    '''
    #p_sum = np.sum(qprob_map.astype(np.float32))
    #prange = np.arange(len(qprob_map))
    pprob = qprob_map[qprob_map>0]
    p_sum = np.sum(pprob.astype(np.float128))
    if len(pprob) == 0:
        print("endmember has collapsed")
        print(qprob_map.max())
        return [0]
    prange = np.arange(len(qprob_map))[qprob_map>0]
    pix = list(np.sort(np.random.rand(number_of_pixels)))
    tally = 0
    pixlist = []
    #print(qprob_map.shape, pprob.shape, pprob.max(), p_sum)
    pixel = pix.pop(0)
    i = 0
    while (pixel >= 0):
        tally += pprob[i]/p_sum
        #print(tally, pixel, i)
        try:
            while pixel < tally:
                pixlist.append(prange[i])
                pixel = pix.pop(0)
        except IndexError:
            pixel = -1.0
        i += 1

    return pixlist

def prob(x, beta):
    return np.exp(-x*beta)

def beta_update(xs, beta_0, alg_type="expectation", tol=0.1):
    ep = xs[xs > 0]
    en = xs[xs <= 0]
    err = 1
    beta_min = 1e-12
    beta_max = 1e6
    if (len(en)==1)&(alg_type=="expectation"):
        beta_0 = beta_max
        return beta_0
    
    
    if len(ep)<=len(en):
        beta_0 = beta_min
        return beta_0
        
    if alg_type=="expectation":
        sp = np.sum(ep*prob(ep.astype(np.float64), beta_0))
        sn = np.sum(-en*prob(en.astype(np.float64), beta_0))
    else:
        sp = np.sum(prob(ep.astype(np.float64), beta_0))
        sn = np.sum(prob(en.astype(np.float64), beta_0))
    sign = np.sign(sp/sn-1)
    pre = 10.0
    while err > tol:
        if sign==np.sign(sp/sn-1):
            pass
        else:
            sign *= -1
            pre = np.sqrt(pre)
        beta_1 = beta_0 * pre**sign
        #print(ep)
        #print(en)
        if alg_type=="expectation":
            sp = np.sum(ep*prob(ep.astype(np.float64), beta_1))
            sn = np.sum(-en*prob(en.astype(np.float64), beta_1))
        else:
            sp = np.sum(prob(ep.astype(np.float64), beta_1))
            sn = np.sum(prob(en.astype(np.float64), beta_1))
        diff = sp/sn - 1 
        err = np.abs(diff)
        #print(beta_1, diff, sp, sn, "sums", alg_type, len(ep), len(en))
        beta_0 = beta_1
        if beta_0 < beta_min:
            return beta_min
        if beta_0 > beta_max:
            return beta_max
    return beta_0

def percentage_mixed_pixels(abus):
    start = abus[abus>0]
    return np.sum(start<1)/len(start)

def quick_nn(image_2D, k_size = 3):
    indices = np.zeros(image_2D.shape[:2], dtype=int)
    args = np.arange(0, image_2D.shape[0]*image_2D.shape[1])
    args = args.reshape(image_2D.shape[:2])
    for i in range(image_2D.shape[0]):
        for j in range(image_2D.shape[1]):
            c = np.arange(np.maximum(0,i-k_size), 
                          np.minimum(i+k_size+1, image_2D.shape[0]))
            d = np.arange(np.maximum(0,j-k_size), np.minimum(j+k_size+1, image_2D.shape[1]))
            
            idxs = [[k,l] for k in c for l in d]
            
            idxs = np.transpose(idxs)
            image_2D[idxs[0], idxs[1]]
            
            diffs = np.sum((image_2D[i,j] - image_2D[idxs[0], idxs[1]])**2, axis=-1)
            sorted_idxs = np.argsort(diffs)
            k = 1
            #print(diffs[sorted_idxs])
            while (diffs[sorted_idxs[k]]<=0):
                k += 1
                if k==len(sorted_idxs):
                    break
                
            if k == len(diffs):
                indices[i,j] = -1
            else:
                idx = sorted_idxs[k]

                idx_full = np.ravel_multi_index(idxs[:,idx], image_2D.shape[:2])
                indices[i,j]=int(idx_full)
    return indices

def align_spectra(gt_map, S):
    '''
    match a set of ground truth maps to a set of unmixing components
    '''
    interact_matrix = np.zeros((len(S),len(gt_map)))
    for i in range(len(S)):
        for j in range(len(gt_map)):
            intersection = np.sum(np.minimum(S[i]**2, gt_map[j]**2))
            union = np.sum(np.maximum(S[i]**2, gt_map[j]**2))
            IoU = intersection/union
            interact_matrix[i, j] = 1-np.dot(S[i]**2,gt_map[j]**2)/np.sqrt(np.dot(S[i]**2,S[i]**2**2)*np.dot(gt_map[j]**2,gt_map[j]**2**2))

            #np.sum((S[i]-gt_map[j])**2)/np.mean([(S[i]**2).sum(),(gt_map[j]**2).sum()])
            
    plt.imshow(interact_matrix.T)
    plt.show()
    row, col = opt.linear_sum_assignment(interact_matrix)
    return row, col
    