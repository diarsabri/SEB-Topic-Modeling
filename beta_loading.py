import numpy as np

def mock_data_beta_loading(self):
    textual_factors = [
        (["hej", "tjena", "hejsan"], [0.2, 0.5, 0.3], 5),

        (["kväll", "natt", "eftermiddag", "morgon", "kvällstid", "imorgon", "idag"], [0.05, 0.05, 0.1, 0.05, 0.15, 0.5, 0.1], 2),

        (["pengar", "dollar", "kronor", "sedlar", "mynt", "valör", "monopol", "kapital"], [0.2, 0.1, 0.1, 0.2, 0.09, 0.01, 0.2, 0.1], 5)
    ]
    
    # Just for if I change the data, so I don't screw something up
    for factor in textual_factors:
        Si = factor[0]
        Fi = factor[1]
        di = factor[2]
        assert len(Si) is len(Fi)
    
    new_doc = {
        # In cluster 1
        "hej": 12,
        "tjena": 1,
        # Not in cluster 1
        "goddag": 5,

        # In cluster 2
        "idag": 1,
        # Not in cluster 2
        "torsdag": 3,
        "måndag": 2,

        # In cluster 3
        "pengar": 13,
        "mynt": 2,
        "kapital": 1,
        # Not in cluster 3
        "money": 9,
        "deg": 7,
    }

    return textual_factors, new_doc


class Beta_Loading():
    def __init__(self, textual_factors):
        self.textual_factors = textual_factors

    def beta_loading(self, dt_sub_vector):
        # textual_factor: triplet (Si, Fi, di), where:
        # Si: words in cluster i - 1 x N
        # Fi: real-value vector representation of cluster i - 1 x N
        # di: Factor importance of cluster i
        # dt_sub_vector: Freq count of every word in the given document, subsections of this will be 
        # used as N_si for each cluster i

        # Find each factor's beta loading
        beta_loadings = []
        for factor in self.textual_factors:
            Si = factor[0]
            Fi = factor[1]

            # Take dt_sub_vector and Si, and create N_si, which is a vector keeping track of the frequencies
            # of each word in the given cluster for the new document. If a word does not appear in the new
            # document, then the freq is 0
            N_si = [0 if word not in dt_sub_vector.keys() else dt_sub_vector[word] for word in Si]

            # Once we have the freqs for the words in cluster i, evaluate the projected beta_loading
            numerator = np.dot(N_si, Fi)
            denominator = np.dot(Fi, Fi)
            beta = numerator / denominator

            beta = beta/float(Fi.shape[0])

            beta_loadings.append(beta)
        
        return beta_loadings

