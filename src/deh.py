import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.optimize as opt
import sklearn.svm as svm
import tables as tab
import src.DR.nmf as nmf
import time  
import copy
import collections

def flat(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flat(i)]
    else:
        return [x]

class Node():
    def __init__(self, spatial_map, classifier, status=True):
        self.map = spatial_map
        self.classifier = classifier
        #elf.splitter = None
        #self.status=status # True = open, False = closed

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
        self.NR = not no_negative_residuals # negative residuals
        self.subfit_parameters = subfit_params
        self.remainder_id = 'none'
        self.splitting_size = 100000
        self.max_depth = 10
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
        self.verbose = False
        self.Cnorm = 1
        self.max_iter = 200
        self.level_scaling = 1.5
        self.beta = 0.1 # rate of grad descent
        self.node_record = {}
        #self.sf = 253
        #return self
            
    def random_init(self, data, n_starting_pts):
        self.parameter_initialization(data)
        starting_pix = np.random.choice(len(data), n_starting_pts)
        groups = [[[i],0] for i in starting_pix]
        
        while len(groups) > 1:
            groups = regroup_1it(data, self.wf(data), groups)
            
        ng = groups[0][0]
        self.nodes[''] = Node(spatial_map=np.ones(n_starting_pts),
                                  classifier=w_avg(data[flat(ng)],
                                                   self.wf(data[flat(ng)])))
        self.nodes[''].origin_pix = ng
        
        while self.nodes_2_include():
            self.add_group_layer(data, self.weights)
            
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
        test1 = svm.LinearSVC(max_iter=100000, dual=True)
        for n in self.nodes:
            if n+'1' in self.nodes:
                points = flat(self.nodes[n+'0'].origin_pix) + flat(self.nodes[n+'1'].origin_pix)
                X = dat[points]
                y = np.zeros(len(points))
                y[len(flat(self.nodes[n+'0'].origin_pix)):]=1
                test1.fit(X,y)
                self.nodes[n].splitter = test1.coef_[0], test1.intercept_
        
    def trim(self, level):
        nodes = list(self.nodes.keys())
        for n in nodes:
            if len(n) > level:
                del self.nodes[n]        
        
    def check_splitting(self):
        return (len(self.nodes_to_split())>0) and (self.get_depth()<self.max_depth)
        
    def parameter_initialization(self, image):
        # takes the image in the ordinary (e.g. N x b), rather than nmf-style 
        self.delta = image.sum(axis=1).mean()
        try:
            self.weights = self.wf(image)
        except AttributeError:
            self.set_weight_function()
            self.weights = self.wf(image)
        self.nodes[''] = Node(spatial_map = np.ones(image.shape[0], dtype=bool),
                              classifier = np.average(image,axis=0,weights=self.weights),
                              status=False)

            
    def set_weight_function(self):
        def wf(image):
            return 1/np.sum(np.abs(image)**self.weight_power, axis=1)
        self.wf = wf
    
    
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
          
        f
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
    
    def simple_predict(self, image):
        #if image.shape[0] != self.weights.shape[0]:
        self.weights = self.wf(image)
        nodes = list(self.nodes.keys())
        nodes.sort()
        self.nodes[''].lmda = np.ones(image.shape[0])
        for n in nodes:
            if n + '1' in nodes: #self.nodes[n].map = np.ones(image.shape[0])
                lmda = classify_from_partition(image, self.nodes[n].splitter[0],
                                    self.nodes[n].splitter[1])
                self.nodes[n+'1'].lmda = lmda
                self.nodes[n+'0'].lmda = 1-lmda
            elif n+'0' in nodes:
                self.nodes[n+'0'].lmda = np.ones(image.shape[0])
        self.nodes[''].map = np.ones(image.shape[0])
        self.lmda_2_map()
        
        S = []
        self.get_end_nodes()
        for i in self.end_nodes:
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
        
    def get_depth(self):
        my = list(self.nodes.keys())
        my.sort(key=len)
        self.depth = len(my[-1])
        return self.depth
    
    def remainder_at_level(self, data, level):
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
        print(all_nodes)
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
            for i in all_nodes:
                if (node in i):
                    if i[:len(node)]==node:
                        self.nodes.pop(i)
        
        print(self.nodes.keys())
        
            
    
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
            if self.nodes[node].map.sum() > self.splitting_size:
                splitting_nodes.append(node)
        return splitting_nodes
    
    def add_another_node_layer_simple(self, data):
        to_split = self.nodes_to_split()
        level = self.get_depth()
        lowest_nodes = [i for i in self.nodes if len(i)==level]
        #print(lowest_nodes)
        eL = self.remainder_at_level(data, level)    
        for i in lowest_nodes:
            if i in to_split:
                print("splittting ", i)
                pos = np.ones(len(eL), dtype=bool)#np.min(self.nodes[i].classifier - eL, axis=1) > 0
                big_err = np.argmax(np.sum(eL[pos]**2, axis=1))
                args = np.argsort(((data - self.nodes[i].classifier)**2).sum(axis=1))
                self.nodes[i].splitter = classifiers_2_svm(self.nodes[i].classifier - eL[pos][big_err],
                                                                   self.nodes[i].classifier + eL[pos][big_err])
                self.nodes[i+'0'] = Node(classifier=self.nodes[i].classifier, spatial_map=np.ones(data.shape[0]))
                self.nodes[i].splitter = (0.001*self.nodes[i].splitter[0], 0)
                self.nodes[i+'1'] = Node(classifier=self.nodes[i].classifier, spatial_map=np.ones(data.shape[0]))
                
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

    def display_level(self, level):
        count=0
        for i in self.nodes:
            if len(i)==level:
                count += 1
        print(count)

        fig, ax = plt.subplots(count, figsize=(8,2*count))
        for i, a in enumerate(ax):
            a.set_xticks([])
            a.set_yticks([])

        counter = 0
        for i in self.nodes:
            if len(i)==level:
                ax[counter].imshow(np.rot90(self.nodes[i].map.reshape(self.plot_size)),
                                   aspect=self.plot_aspect, vmin=0, vmax=1, interpolation='nearest')
                ax[counter].set_ylabel(self.nodes[i].map.sum())
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
            
            
    def save(self, filename, title='Deep Endmember Hierarchy', save_labels=()):
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
        self.h5file = tab.open_file(filename, mode='a')
        table = self.h5file.root.nodes
        for row in table.iterrows():
            node = row['name'].decode('ascii')
            self.nodes[node] = Node(None, row['classifier'])
            if np.sum(np.abs(row['splitter_w'])) > 0:
                self.nodes[node].splitter = (row['splitter_w'], row['intercept']) 
                
    def display_spectra(self, spectra_list, names = (), wl=(), normalizer = 1, **kwargs):
        if len(wl)==0:
            wl = np.arange(len(self.nodes[''].classifier))
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
                  n_update_points=0, attenuation=1):
        
        self.update_from_level_S_V(data, beta=beta, alg=alg, n_update_points=n_update_points, attenuation=attenuation)

        #if alg=='complex':
        #    self.populate_fns( data, target='S', alg=alg)
        for n in self.nodes:
            self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
                                              dtype=np.float32)

        if max_level == -1:
            max_level = self.get_depth()
        #if alg=='complex':
        #    for i in range(max_level-2):
        #        self.update_from_level_S_I(i+2, data, scaling_factor=scaling_factor)
        
        if n_update_points > 0:
            update_pix = np.random.choice(len(data), n_update_points)
            for i in range(max_level-1):
                self.update_from_level_S_II(i+1, data[update_pix], scaling_factor=scaling_factor)
        else:
            for i in range(max_level-1):
                self.update_from_level_S_II(i+1, data, scaling_factor=scaling_factor)
         
        scaling_factor = np.mean(1/self.weights)
        long_nodes = [n for n in self.nodes if len(n) >= up_level]
        for n in long_nodes:
            try:
                step = (beta)*scaling_factor*self.nodes[n].S_update
                step_mag = np.sum(np.abs(step))
                #print(step_mag)
                if step_mag / np.sum(self.nodes[n].classifier) > max_step_r:
                    step = step / step_mag * max_step_r
                self.nodes[n].classifier += step
            except AttributeError:
                pass

        for n in self.nodes:
            try:
                self.nodes[n].classifier[self.nodes[n].classifier<0]=0
                del self.nodes[n].fns
            except AttributeError:
                pass
            
    def lowest_level_S(self, data, beta=0.1, alg='complex'):
        d = self.get_depth()
        for n in self.nodes:
            self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
                                                  dtype=np.float32)
        self.update_from_level_S_V(data, beta=beta, alg=alg)
        scaling_factor = np.mean(1/self.weights)
        for n in self.nodes:
            try:
                self.nodes[n].classifier += beta*scaling_factor*self.nodes[n].S_update
                self.nodes[n].S_update = np.zeros(len(self.nodes[n].classifier),
                                                  dtype=np.float32)
                self.nodes[n].classifier[self.nodes[n].classifier<0]=0
            except AttributeError:
                pass
            
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
                             attenuation = 1):
        # only set up to work on the lowest level
        # Archetypal -analysis update
        # dependence on beta removed via line search
        start = time.time()
        level = self.get_depth()
        if alg=='complex':
            self.predict(data)
        elif alg=='simple':
            self.simple_predict(data)
        incl_nodes = [n for n in self.nodes if len(n) == level]
        #incl_classifiers = copy.deepcopy([self.nodes[n].classifier for n in incl_nodes])
        #data_stack = np.vstack([incl_classifiers, data])
        if n_update_points>0:
            occs = {}
            for n in self.nodes:
                occs[n] = self.nodes[n].map 
                    
        eL = self.remainder_at_level(data, level)
        #for pix in data:S
        #if '1' in incl_nodes:
        #    incl_nodes = ['1']
        for n in incl_nodes:
            if n_update_points > 0:
                om = occs[n]
                #om = om > 0
                #update_pix = np.random.choice(om.sum(), np.minimum(n_update_points, np.sum(om)))
                update_pix = rand_sel(om, n_update_points)
                ldata = data[update_pix]
                #ldata = data[om][update_pix]
                self.simple_predict(ldata)
            else:
                ldata = data
            eL = self.remainder_at_level(ldata, level)
            lc = self.nodes[n].classifier
            rem = ldata-lc # to throw errors if there is a bug
            #eXdelta = rem@eL.T
            if n_update_points > 0:
                elo_sum = np.sum(np.multiply((self.weights).T, eL.T), axis=1)
            else:
                elo_sum = np.sum(np.multiply((self.weights*self.nodes[n].map).T, eL.T), axis=1)
            elo = elo_sum@rem.T
            #norm = np.array([i@i for i in rem])
            #elo = np.multiply((self.weights*self.nodes[n].map).T,eXdelta)#.T@rem.T
            #elo_sum = elo.sum#eld = np.array([elo[i,i] for i in range(len(elo))])
            if n_update_points > 0:
                denom_2_sum = self.weights * self.nodes[n].map + 1e-16
            else:
                denom_2_sum = self.weights * self.nodes[n].map**2 + 1e-16
            #denom_adj = self.weights[i] * self.nodes[n].map[i]**2
            denom = np.sum(denom_2_sum)
            norm = np.array([i@i for i in rem]) +1e-16
            #self_int = np.array([self.weights[i]*self.nodes[n].map[i]*rem[i]@eL[i] for i in range(len(rem))])
            #print(len(norm),'norm')
            #slope = np.sum((eL.T*self.weights*self.nodes[n].map).T@rem.T, axis=1)
            a = (elo)/(denom*norm)#np.sum(elo-np.diag(eld), axis=1)/(denom*norm)
            a_keep = ((a<1) & (a>0)) #& (slope < 0)
            a = np.maximum(np.minimum(a, 1),0)
            #a_plus = a > 0
            #print(elo.shape)
            #prefactor_a = norm*np.sum(self.nodes[n].map**2*self.weights)
            total_change = - (denom*norm)/2*a**2 + elo.T*a.T
            #np.sum((elo.T*a).T, axis=1)
            #np.sum(-prefactor_a.T/2*a[a_keep]**2 + elo[a_keep].T*a[a_keep], axis=0)
            #print(len(total_change), 'tc')
            #print(a[a_keep])
            #print(total_change / len(data))
            
            best_sig = np.argmin(a)
            best_sig2 = np.argmax(total_change)
            
            if total_change[best_sig2] > 0:
                top_sum = a[best_sig2]
                #print('1', total_change[best_sig], total_change[best_sig2], slope[best_sig2])
                #print(n, a)
                #if slope[best_sig2] < 0:
                # = self.weights * self.nodes[n].map**2 + 1e-16
                #print(top_sum, np.sum(denom_2_sum))
                beta = top_sum #/ np.sum(denom_2_sum)
                beta = np.minimum(np.maximum(beta,0),1)
                beta /= attenuation
                print(n, beta, time.time() - start)
                classifier_new = self.nodes[n].classifier + beta*rem[best_sig2]
                #eL += np.outer(self.nodes[n].map, beta*rem[best_sig])
                self.nodes[n].classifier = classifier_new
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
            #self.nodes[n].classifier = classifier_new
            

            
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
        delta = 1
        obj_orig, o_scores = evaluate()
        while delta > tol:
            update_pix = np.random.choice(len(data), n_update_points)
            self.lowest_level_S(data[update_pix],beta=beta, alg=alg)
            update_pix = np.random.choice(len(data), n_update_points)
            self.one_step_cyclic(data, lowest=True, n_update_points=n_update_points)
            if record_weights:
                self.append_node_record()
            new_obj, o_scores = evaluate()
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
                      obj_record=(), up_level=0, scaling_factor=2, alg='complex', record_weights=False):
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
        delta = 1
        obj_orig, o_scores = evaluate()
        while delta > tol:
            #update_pix = np.random.choice(len(data), np.minimum(len(data), n_update_points))
            #self.one_step(data[update_pix],beta=beta, up_level=up_level, scaling_factor=scaling_factor, alg=alg)
            self.one_step_S(data,beta=beta, up_level=up_level, scaling_factor=scaling_factor, alg=alg,
                           n_update_points = n_update_points, attenuation=1)
            #update_pix = np.random.choice(len(data), n_update_points)
            #self.one_step_cyclic(data[update_pix], scaling_factor=scaling_factor)
            self.one_step_cyclic(data, scaling_factor=scaling_factor, n_update_points=n_update_points)
            if record_weights:
                self.append_node_record()
            new_obj, o_scores = evaluate()
            delta = np.abs((obj_orig - new_obj)/obj_orig)
            #delta = 0.5*delta + 0.5*halfdelta
            obj_orig = new_obj
            obj_record.append([o_scores[-1],3, self.get_depth(), o_scores[0], o_scores[1]])
            print(new_obj,o_scores)

            
            
            
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
        eL = self.remainder_at_level(image, 0)    
        pos = np.min(self.nodes[''].classifier - eL, axis=1) > 0
        big_err = np.argmax(np.sum(eL[pos]**2, axis=1))
        args = np.argsort(((image - self.nodes[''].classifier)**2).sum(axis=1))
        self.nodes[''].splitter = classifiers_2_svm(self.nodes[''].classifier - eL[pos][big_err],
                                                                   self.nodes[''].classifier + eL[pos][big_err])
        self.nodes['0'].classifier = self.nodes[''].classifier
        self.nodes['1'].classifier = self.nodes[''].classifier
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
            delta = 0 
            print(delta)
            while delta < 0.95:
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
                delta = obj_record[-1][-1]
                #print(np.abs(old_obj - new_obj)/old_obj)
            
            if alg=='complex':
                self.predict(image)
            elif alg=='simple':
                self.simple_predict(image)
            self.display_level(self.get_depth())
            
    def switch_training(self, image, beta, tol, n_update_points=1000, scaling_factor=4,
                        obj_record=(), sampling_points=(), alg='complex'):
        self.quick_alt_ll( image, beta=beta, tol=tol,
                               n_update_points=n_update_points,
                               obj_record=obj_record, sampling_points=sampling_points, alg=alg)
        old_obj = 1
        new_obj = 0
        while np.abs(old_obj - new_obj) > tol:
                self.quick_alt( image, beta=beta, tol=tol,
                      n_update_points=n_update_points,
                            sampling_points=sampling_points, obj_record=obj_record, scaling_factor=scaling_factor, alg=alg)
                self.quick_alt_ll( image, beta=beta, tol=tol,
                               n_update_points=n_update_points,
                               obj_record=obj_record, sampling_points=sampling_points, alg=alg)
                old_obj = new_obj
                new_obj = obj_record[-1][0]
                
            
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
        
        
    def aggregate_node_at_level(self, node, level, data):
        top_level = len(node)
        out = np.zeros_like(data)
        nodes_in_level = [i for i in self.nodes if len(i)==level]

        for snode in nodes_in_level:
            if snode[:(top_level)]==node:
                #switch out for lambda
                if snode[top_level]=='0':
                    loc_lambda = self.intermediate_lambda_product(node + '0', snode)
                    out += np.outer(loc_lambda, self.nodes[snode].classifier)
                elif snode[top_level]=='1':
                    loc_lambda = self.intermediate_lambda_product(node + '1', snode)
                    out -= np.outer(loc_lambda, self.nodes[snode].classifier)

        return out
    
    def node_grad(self, node, data, scaling_factor=2, var='W', metropolis=False):
        self.simple_predict(data)
        
        #self.populate_fns(data, alg='simple')
        depth = self.get_depth()
        eL = self.remainder_at_level(data, depth)
        print('err', np.sum(np.multiply((eL**2).T, self.weights), axis=0).mean())
        #for i in range(depth):
        #    #print("updateing", i)
        #    self.update_from_level(i+1, data, scaling_factor=scaling_factor, alg='simple')
        
        factors = 1.0 * scaling_factor**np.arange(depth + 1)
        factors /= np.sum(factors)
        
        init_incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
        
        if var=='W':
            #print(node, var, ' include ', np.sum(init_incl))
            num = np.zeros(data.shape) 
            #factor_sum = 0
            for i in range(len(node)+1,depth+1):
                eL = self.remainder_at_level(data, i)
                out = self.aggregate_node_at_level(node, i, data)
                if metropolis:
                    werr = np.sum(np.multiply(eL, out), axis=1)
                else:
                    werr = np.sum(np.multiply(eL, out), axis=1)*self.nodes[node].map
                werr_w = np.multiply(self.weights, werr)
                #nl = np.multiply(self.weights, np.multiply(np.sum(out**2, axis=1), xw**2))
                num += factors[i]*np.multiply(data.T, werr_w).T
                #factor_sum += factors[i]
                #denom += factors[i]*nl*self.nodes[node].map**2
            #plt.plot(np.mean(num[init_incl], axis=0)/np.sum(np.mean(num[init_incl], axis=0)))
            #plt.plot(self.nodes[node].W_update/np.sum(self.nodes[node].W_update),'--')
            #plt.show()
            self.nodes[node].W_update = -np.mean(num[init_incl], axis=0)

        if var=='W':
            #print(self.nodes[node].W_update)
            xw = data@self.nodes[node].W_update/2
        elif var=='h':
            xw = 0.5
            
        #print(np.mean(np.abs(xw)),'xw')

        num = np.zeros(data.shape[0])
        denom = np.zeros(data.shape[0])

        factors = 1.0 * scaling_factor**np.arange(depth + 1)
        factors /= np.sum(factors)
        

        for i in range(len(node)+1,depth+1):
            eL = self.remainder_at_level(data, i)
            out = self.aggregate_node_at_level(node, i, data)
            werr = np.sum(np.multiply(eL, out), axis=1)
            werr_w = np.multiply(np.multiply(self.weights, werr), xw)
            nl = np.multiply(self.weights, np.multiply(np.sum(out**2, axis=1), xw**2))
            if metropolis:
                num += factors[i]*werr_w#*self.nodes[node].map
                denom += factors[i]*nl*self.nodes[node].map
            else:    
                num += factors[i]*werr_w*self.nodes[node].map
                denom += factors[i]*nl*self.nodes[node].map**2

        
        if np.sum(init_incl)==0:
            return -1
        pl = (data@self.nodes[node].splitter[0] + self.nodes[node].splitter[1] +1)/2
        int_level1 = (1-pl)/(xw)
        int_level0 = (-pl)/(xw)
        intercepts = np.minimum(int_level1, int_level0)
        beta = -np.sum(num[init_incl])/np.sum(denom[init_incl])
        
        
        if var=='h':
            beta_flag=False
            if beta < 0:
                beta = -beta
                int_level0 = -int_level0
                int_level1 = -int_level1
                num = -num
                beta_flag = True
                

        int_level0[int_level0<=0]=10**16
        int_level1[int_level1<=0]=10**16
        intercepts = np.minimum(int_level1, int_level0)
        
        print('veta,',beta, intercepts.min())
        if beta > intercepts.min():
            beta = find_opt_beta(intercepts, init_incl, num, denom)
            print('beta,', beta)

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
           
            
        if var=='W':
            splitter = self.nodes[node].splitter[0] + beta*self.nodes[node].W_update, self.nodes[node].splitter[1]
        if var=='h':
            splitter = self.nodes[node].splitter[0], self.nodes[node].splitter[1] + beta

        old_splitter = self.nodes[node].splitter      
        self.nodes[node].splitter = splitter
        for n in self.nodes:
            try:
                self.nodes[n].W_update[:]=0
                self.nodes[n].h_update=0
            except AttributeError:
                pass
            
        self.simple_predict(data)
        eL = self.remainder_at_level(data, depth)
        print(node, var, ' include ', np.sum(init_incl))
        
        print('err', np.sum(np.multiply((eL**2).T, self.weights), axis=0).mean())
        
        init_incl = np.abs(self.nodes[node+'0'].lmda - 0.5) < 0.5
        if np.sum(init_incl)==0:
            self.nodes[node].splitter = old_splitter
            return -1

        
    def one_step_cyclic(self, data, scaling_factor=2, n_update_points=0, lowest=False):
        nodes = list(self.nodes.keys())
        depth = self.get_depth()
        if lowest:
            start=depth-1
        else:
            start=0
        if n_update_points > 0:
            self.simple_predict(data)
            prop_map = {}
            for n in nodes:
                if len(n) >= 0:
                    prop_map[n] = self.nodes[n].map
                    #prop_map[n] = self.nodes[n].map>0
        for level in range(depth-1, start-1, -1):
            for n in nodes:
                if len(n)==level:
                    if n+'1' in nodes:
                        self.nodes[n].h_update = 0 
                        self.nodes[n].W_update = np.zeros(len(self.nodes[n].classifier),
                                              dtype=np.float32)
                        #plt.plot(self.nodes[n].splitter[0])
                        if n_update_points > 0:
                            pm = prop_map[n] #> 0
                            pm_max = np.minimum(prop_map[n+'0'].max(), prop_map[n+'1'].max())
                            print(pm_max, 'pmmax', n)
                            if pm_max > 0.1:
                                update_pix = rand_sel(pm, n_update_points)
                                #update_pix = np.random.choice(np.sum(pm),
                                #                              np.minimum(np.sum(pm), n_update_points))
                                self.node_grad(n, data[update_pix], scaling_factor=scaling_factor, var='W',
                                              metropolis=True)
                                self.node_grad(n, data[update_pix], scaling_factor=scaling_factor, var='h',
                                              metropolis=True)
                            else:
                                self.nodes[n].splitter = [self.nodes[n].splitter[0]*0,
                                                          self.nodes[n].splitter[1]*0]
                        else:
                            self.node_grad(n, data, scaling_factor=scaling_factor, var='W')
                            self.node_grad(n, data, scaling_factor=scaling_factor, var='h')
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
        for l in errors:
            print(l, errors[l])
        return errors
            
    
    
def find_opt_beta(intercepts, initial_include, num_weights, denom_weights):
    '''
    Approximate the beta for which the slope vanishes
    '''
    EPS = 1e-16
    
    old_beta = 0
    num = np.sum(num_weights[initial_include])
    denom = np.sum(denom_weights[initial_include])
    args = np.argsort(intercepts)
    slope = num
    #print(slope)
    i = 0
    shift = 0
    beta = 0
    min_int = np.sum(initial_include)
    while slope < 0:
        old_beta = beta
        old_slope = slope
        ID = args[i]
        beta = intercepts[ID]
        deltabeta = beta - old_beta
        shift += deltabeta*denom
        initial_include[ID] = ~initial_include[ID]
        min_int = np.minimum(min_int, np.sum(initial_include))
        if min_int < 1:
            slope = 0
            beta = old_beta
        else:
            if initial_include[ID]:
                new_num = num + num_weights[ID]
                denom_new = denom + denom_weights[ID]
            else:
                new_num = num - num_weights[ID]
                denom_new = denom - denom_weights[ID]

            num = new_num
            denom= denom_new
            slope = new_num + shift
        i += 1
        if i==len(args):
            slope=0
    print("min ints", min_int) 
    return np.minimum(np.maximum(beta-EPS,0), np.maximum(old_beta-old_slope/denom,0))#(old_beta + beta) / 2

            
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
    p_sum = np.sum(qprob_map)
    #prange = np.arange(len(qprob_map))
    pprob = qprob_map[qprob_map>0]
    if len(pprob) == 0:
        print("endmember has collapsed")
        print(qprob_map.max())
        return -1
    prange = np.arange(len(qprob_map))[qprob_map>0]
    pix = list(np.sort(np.random.rand(number_of_pixels)))
    tally = 0
    pixlist = []
    #print(pprob.shape, pprob)
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