from rhp import RHP
import json
import numpy as np
import pandas as pd

class Cluster():
    def __init__(self, w2v_model, w2v_index, rhp):
        self.cluster_map = None
        self.w2v_model = w2v_model
        self.w2v_index = w2v_index
        self.pairs = None
        self.rhp = rhp
        self.removed_word = None
        self.distances_pairs = pd.DataFrame(columns = ['Word_1', 'Word_2', 'Distance'])
        self.cluster_index = 0


    def create_initial_clusters(self):
        vocabulary = self.w2v_model.wv.vocab.keys()
        self.pairs = {word : {'words': [word], 'center': self.w2v_model.wv[word].tolist()} for word in vocabulary}
    
    def find_closest_pair(self, cluster=None):
        # First iterate over all combinations of pairs and save to as structure {word_1: pair_words: [word_2], distances:[] }, this way we can sort the keys to find the closest distances.
        if cluster is not None:
            # returns it self and then the closest, hence index 1
            top_candidates, _ = self.w2v_index.find_candidates(
                                    self.pairs[cluster]['center'], max_candidates=4, key_dist=1, dist_func=self.rhp.cosine_dist)
            if len(top_candidates) <= 1:
                return True
            neighbour, distance = top_candidates[1]
            self.distances_pairs = self.distances_pairs.append({'Word_1':cluster, 'Word_2': self.w2v_model.wv.index2word[neighbour], 'Distance': distance}, ignore_index=True)
        else:
            for word in self.pairs.keys():
                # returns it self and then the closest, hence index 1
                top_candidates, _ = self.w2v_index.find_candidates(
                                        self.pairs[word]['center'], max_candidates=4, key_dist=1, dist_func=self.rhp.cosine_dist)
                #print(word, ': ', top_candidates)
                if len(top_candidates) <= 1:
                    continue
                neighbour, distance = top_candidates[1]
                self.distances_pairs = self.distances_pairs.append({'Word_1':word, 'Word_2': self.w2v_model.wv.index2word[neighbour], 'Distance': distance}, ignore_index=True)
            print('------------Finished clalculating all pairs------------')
                
        #print(word)
        #print(neighbour)

        #print(self.w2v_model.wv.index2word[neighbour])
        #print(distance)
        
        self.distances_pairs = self.distances_pairs.sort_values(by=['Distance'])
        # print(self.distances_pairs.head())
        '''candidate_w = self.distances_pairs.iloc[0]['Word_2']
        top_candidates, _ = self.w2v_index.find_candidates(
                                    self.pairs[candidate_w]['center'], max_candidates=4, key_dist=1, dist_func=self.rhp.cosine_dist)
        print(candidate_w)
        print(top_candidates)
        neighbour, distance = top_candidates[1]
        res_word = self.w2v_model.wv.index2word[neighbour]
        print("The closest nb to: ", candidate_w, " is: ", res_word, " with distance: ", distance)'''
        
        return False
        
    
    def cluster_closest(self):
        # Cluster the closest
        #print(self.distances_pairs.head())
        #print(len(self.distances_pairs))
        closest_pair = self.distances_pairs.iloc[0]
        #print('---------------------------------')
        #print('Closest pair is: ', closest_pair)
        #print('---------------------------------')
        cluster_name = 'cluster'+str(self.cluster_index)
        self.cluster_index+=1
        
        if cluster_name not in self.pairs.keys():
            self.pairs[cluster_name] = {'words': [], 'center': None}

        cluster_1_words = self.pairs[closest_pair['Word_1']]['words']
        cluster_2_words = self.pairs[closest_pair['Word_2']]['words']

        self.pairs[cluster_name]['words'].extend(cluster_1_words)
        self.pairs[cluster_name]['words'].extend(cluster_2_words)
        
        #print("cluster name: ", cluster_name)
        #print(cluster_name in self.pairs.keys())
        #print(closest_pair)
        new_center = np.mean([np.array(self.pairs[closest_pair['Word_1']]['center']), np.array(self.pairs[closest_pair['Word_2']]['center'])], axis=0).tolist()
        self.pairs[cluster_name]['center'] = new_center
        
        word_emb_1_to_del = self.pairs[closest_pair['Word_1']]['center']
        word_emb_2_to_del = self.pairs[closest_pair['Word_2']]['center']
        
        # Delete the clusters from the pairs dict
        del self.pairs[closest_pair['Word_1']]
        del self.pairs[closest_pair['Word_2']]
        
        # Delete from dataframe
        word_1_idecies = self.distances_pairs.index[self.distances_pairs.Word_1 == closest_pair['Word_1']]
        self.distances_pairs = self.distances_pairs.drop(word_1_idecies)

        word_2_idecies = self.distances_pairs.index[self.distances_pairs.Word_1 == closest_pair['Word_2']]
        self.distances_pairs = self.distances_pairs.drop(word_2_idecies)

        word_11_idecies = self.distances_pairs.index[self.distances_pairs.Word_2 == closest_pair['Word_1']]
        self.distances_pairs = self.distances_pairs.drop(word_11_idecies)

        word_22_idecies = self.distances_pairs.index[self.distances_pairs.Word_2 == closest_pair['Word_2']]
        self.distances_pairs = self.distances_pairs.drop(word_22_idecies)

        #word_11_idecies = self.distances_pairs.index[self.distances_pairs.Word_2 == closest_pair['Word_1']]
        #word_22_idecies = self.distances_pairs.index[self.distances_pairs.Word_2 == closest_pair['Word_2']]
        
        return cluster_name, word_emb_1_to_del, word_emb_2_to_del
        # Add cluster to dataframe

                              
        # Remove the words from df
        # add cluster to dataframe
        # Remove words from index
        # Add cluster to index
        # Hash cluster center in ind

    
    
    
    def cluster_nodes(self, cluster_limit):
        self.removed_words = {key: True for key in self.pairs.keys()}
        #print(self.pairs.keys())
        num_clusters = len(self.pairs.keys())
        while num_clusters > cluster_limit:
            # print(num_clusters)
            for word in self.pairs.keys():
                if not self.removed_words[word]:
                    continue
                center = self.pairs[word]['center']
                top_candidates, _ = self.w2v_index.find_candidates(
                                    center, max_candidates=2, key_dist=1, dist_func=self.rhp.cosine_dist)
                for word_id, dist in top_candidates:
                    neighbour = self.w2v_model.wv.index2word[word_id]
                    if neighbour != word:
                        if not self.removed_words[neighbour]:
                            continue
                        self.pairs[word]['neighbours'][neighbour] = self.w2v_model.wv[neighbour].tolist()
                        self.pairs[word]['center'] = self.calculate_center(self.pairs[word]['center'], self.w2v_model.wv[neighbour]).tolist()
                        self.removed_words[neighbour] = False
                        num_clusters -=1
                    if num_clusters<=cluster_limit:
                        #print(num_clusters, ' number of cluster')
                        i = 0
                        for key_to_remove in self.removed_words.keys():
                            if self.removed_words[key_to_remove]:
                                del self.pairs[key_to_remove]
                            else:
                                # print(self.removed_words[key_to_remove])
                                i+=1
                        #print(i)
                        return
                    

    def calculate_center(self, old_center, new_word_vector):
        return np.mean([np.array(old_center), np.array(new_word_vector)], axis=0)

    def claculate_distance_for_pairs(self):

        i = 0
        for word in self.pairs.keys():
            #distance_list, _ = self.w2v_index.query(self.w2v_model.wv[word], topk=-1, key_dist=1, dist_func=RHP.cosine_dist)
            #distance_list, _ = self.w2v_index.find_candidates(self.w2v_model.wv[word], max_candidates=len(self.pairs.keys()), key_dist=1, dist_func=RHP.cosine_dist)
            for secondary_word in self.pairs[word].keys():
                #pair_word = self.w2v_model.wv.index2word[word_id]
                if self.pairs[word][secondary_word] is None:
                    dist = self.rhp.cosine_dist(self.w2v_model.wv[word], self.w2v_model.wv[secondary_word])
                    self.pairs[word][secondary_word] = dist
                    self.pairs[secondary_word][word] = dist
                i+=1
                if (i%100 == 0):
                    print('processed ', i, ' words')

    def save_pairs(self, path):
        keys = self.pairs.keys()
        count_words = 0
        count_centers = 0
        
        '''
        for key in keys:
            if len(self.pairs[key]['words']) > 5:
                print('Key - ' + str(key))
                print('Words - Type' + str(type(self.pairs[key]['words'])))
                print(self.pairs[key]['words'])
                print('--------------------------------------------------------------')
                
            # print('Center - Type' + str(type(self.pairs[key]['center'])))
            # print(self.pairs[key]['center'])
            # print('--------------------------------------------------------------')
        '''
        
        new_dict = {word : {'words': self.pairs[word]['words'], 'center': self.pairs[word]['center']} for word in self.pairs.keys()}
        
        with open(path, 'w') as write_file:
            json.dump(new_dict, write_file, indent=4, sort_keys=True)
        print("dumped")
