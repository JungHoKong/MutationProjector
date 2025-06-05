import networkx as nx
from collections import defaultdict
import pandas as pd
import ndex2
import numpy as np
from pathlib import Path
from load_geneList import *




## import network
class load_network():
    def __init__(self):
        super(load_network, self).__init__()
        try: 
            self.fi_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.fi_dir = Path().resolve().parent

            
    def return_edges(self, network, geneList):
        '''
        Provide edge information for given network name and custom gene list
        -----------------------
        # Input
        network : 'SL' (or 'synthetic_lethal'), 'GRN', 'STRING', 'physical_ppi', 'E3', 'phosphorylation', 'DDRAM', 'PCNET'
        gene list : [genes]

        -----------------------
        # Returns
        list of lists containing [gene1 index, gene2 index].
        shape (Num edges, 2)
        '''
        
        ##=================
        # load networks        
        # cx file
        out = defaultdict(list)
        if network.upper() == 'STRING':
            fiName = 'STRING'
        elif network.upper() == 'DDRAM':
            fiName = 'DNA Damage Association Score (DAS) network'
        elif network.upper() == 'PCNET':
            fiName = 'PCNET'
        elif (network.upper() == 'SL') or (network == 'synthetic_lethal'):
            fiName = 'SL'
        elif network.upper() == 'GRN':
            fiName = 'GRN'
        elif network.upper() == 'PHYSICAL_PPI':
            fiName = 'physical_ppi'
        elif network.upper() == 'E3':
            fiName = 'E3_ubiquitination'
        elif network.upper() == 'PHOSPHORYLATION':
            fiName = 'phosphorylation'
            
        net = ndex2.create_nice_cx_from_file(f'{self.fi_dir}/data/networks/{fiName}.cx')
        # add edges
        for edge_id, edge in net.get_edges():
            source = net.get_node(edge['s'])['n']
            target = net.get_node(edge['t'])['n']
            if (source in geneList) and (target in geneList):
                out['gene1'].append(source)
                out['gene2'].append(target)
        out = pd.DataFrame(out, columns=['gene1', 'gene2'])
    
            
        ##=================
        # convert gene IDs
        gene1_list = out['gene1'].tolist()
        gene2_list = out['gene2'].tolist()
        out['gene1'] = convert_geneIDs(gene1_list)
        out['gene2'] = convert_geneIDs(gene2_list)
            
        
        ##=================
        # return edges
        final = []
        G = nx.Graph()
        for val in out.values:
            gene1, gene2 = val
            if gene1 == gene2: continue
            if not gene1 in geneList: continue
            if not gene2 in geneList: continue
            # gene index
            gene1_idx = geneList.index(gene1)
            gene2_idx = geneList.index(gene2)
            
            if G.has_edge(gene1_idx, gene2_idx) == False:
                G.add_edge(gene1_idx, gene2_idx)
        
        for edge in G.edges():
            gene1_idx, gene2_idx = edge
            final.append((gene1_idx, gene2_idx))
            final.append((gene2_idx, gene1_idx))
        
        final = list(set(final))
        final = np.array([list(tmp_list) for tmp_list in final])
        return final
