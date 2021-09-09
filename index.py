'''
This implementation was inpired by https://github.com/dongguosheng/lsh
'''

from operator import itemgetter

from rhp import RHP


class Index(object):
    def __init__(self, lsh):
        self.lsh = lsh
        self.hash_to_word_index = {}  # Keep track of which words hash into the same buckets
        self.word_embedding = []  # Keep track of the embeddings and ids of each word
        self.w_emb_to_index = {} 

    def index(self, w_emb, w_id):
        # Store the word embedding and its word index
        self.word_embedding.append(w_emb)
        self.w_emb_to_index[tuple(w_emb)] = w_id

        # Hash the word embedding multiple times
        hash_list = self.lsh.hash(w_emb)
        for hash_func_id, hash_key in enumerate(hash_list):
            # Add the hash function id to keep track of which function gave the specific hash
            hash_key += ('_' + str(hash_func_id))

            # Add the word id to the hash bucket
            self.hash_to_word_index.setdefault(hash_key, set())
            self.hash_to_word_index[hash_key].add(w_id)

    def delete_index(self, w_emb):
        w_id = self.get_idx(w_emb)
        # Hash the word 
        hash_list = self.lsh.hash(w_emb)
        for hash_func_id, hash_key in enumerate(hash_list):
            # Add the hash function id to keep track of which function gave the specific hash
            hash_key += ('_' + str(hash_func_id))

            # Remove the w_id from the hash bucket
            if hash_key in self.hash_to_word_index.keys():
                self.hash_to_word_index[hash_key].remove(w_id)
        
        # Remove the w_id from the set
        self.w_emb_to_index.pop(tuple(w_emb))

    def get_idx(self, w_emb):
        return self.w_emb_to_index[tuple(w_emb)]
            
    def find_candidates(self, base_w_emb, max_candidates=10, key_dist=1, dist_func=RHP.cosine_dist):
        dist_to_base_word = {}
        if key_dist >= 0:
            # Hash the word we are probing for candidates
            hash_list = self.lsh.hash(base_w_emb)
            for hash_func_id, hash_key in enumerate(hash_list):
                # Find all hash signatures that are at most key_dist bits different from the hash_key
                for candidate_hash in RHP.find_nearby_bit_strings(hash_key, key_dist):
                    # Sort the candidate hashes by hash function
                    candidate_hash += ('_' + str(hash_func_id))
                    if candidate_hash in self.hash_to_word_index:
                        for candidate_w_id in self.hash_to_word_index[candidate_hash]:
                            if candidate_w_id in dist_to_base_word:
                                continue

                            # Calculate the distance between the candidate and base words
                            #print("Length of word embedding list is: ", len(self.word_embedding))
                            #print("Candidate word id is: ", candidate_w_id)
                            dist = dist_func(self.word_embedding[candidate_w_id], base_w_emb)
                            dist_to_base_word[candidate_w_id] = dist
        else:
            # brute force search
            for stored_w_id, stored_w_emd in enumerate(self.word_embedding):
                dist = dist_func(stored_w_emd, base_w_emb)
                dist_to_base_word[stored_w_id] = dist
        #print('Candidates: %d' % len(dist_to_base_word))
        num_candidates = len(dist_to_base_word)
        dist_to_base_word = sorted(dist_to_base_word.items(), key=itemgetter(1), reverse=False)
        return dist_to_base_word[: max_candidates], num_candidates
