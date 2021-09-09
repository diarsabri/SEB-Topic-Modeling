import os
from gensim import models
import pandas
import json
import numpy as np
import nltk 
nltk.data.path.append('./nltk_data')
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from datetime import datetime



from w2v import Word2Vec_routine
from rhp import RHP
from index import Index
from clustering import Cluster
from beta_loading import Beta_Loading, mock_data_beta_loading



class Experiment():
    def __init__(self, training_data, testing_data, vec_size, n_bit, n_table, num_clusters, w2v_path, cluster_path, window, min_count, negative, alpha, min_alpha, sample, google):
        self.training_data = training_data
        self.testing_data = testing_data
        self.vec_size = vec_size
        self.n_bit = n_bit
        self.num_clusters = num_clusters
        self.w2v_path = w2v_path
        self.cluster_path = cluster_path
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = alpha
        self.google = google
        self.sample = sample
        self.w2v_model = None
        self.clusters = None
        self.SVM = None
        self.RF = None
        self.PCA = None
        self.n_table = n_table
        self.SVM_radial = None
        self.w2v_sentences = None
        
        
    def create_w2v_model(self, path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google):
        # Create a w2v with the given training data

        word2vec = Word2Vec_routine(train_data['prepped_total_text_cleaned'])
        word2vec.create_sentence_vec_represenation()
        word2vec.create_embedding(vec_size, window, min_count, negative, alpha, min_alpha, sample, google)
        word2vec.save_model(path)
        self.w2v_sentences = word2vec._article_vector_representation
        

        return word2vec.w2v_model
    
        
    def init_w2v_model(self, path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google):
        '''
        if os.path.exists(path):
            w2v = models.Word2Vec.load(path)
        else:
            w2v = create_w2v_model(path, train_data)
        '''
        w2v = self.create_w2v_model(path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google)
        return w2v

    def cluster_vocab(self, w2v_model, target_num_clusters, path):    
        # To be able to user the code for clusters we need
        n_bit = 16 # Length of hash
        n_dim = w2v_model.wv.vector_size # Word2Vec size
        n_table = 5 # Number of hash funtions (buckets)
        rhplsh = RHP(n_dim, n_bit, n_table)
        rhplsh.init_hyperplane()
        w2v_index = Index(rhplsh)
        idx = 0
        for v in w2v_model.wv.vectors:
            w2v_index.index(v, idx)
            idx += 1
        idx-=1
        #w = 'artificial'
        #w_emb = w2v.wv[w]
        #w_id = w2v.wv.vocab.get(w).index
        #top_candidates, total_candidates = w2v_index.find_candidates(
            #w_emb, max_candidates=30, key_dist=1, dist_func=RHP.cosine_dist)
        #print('Base word: %s' % w)
        #print('Total possible candidates: %d' % total_candidates)


        cluster_limit = target_num_clusters
        clusters = len(w2v_model.wv.vocab.keys())
        cluster = Cluster(w2v_model, w2v_index, rhplsh)
        cluster.create_initial_clusters()

        cluster_name = None
        word_id = idx

        test = True
        print('Initial clusters: ' + str(len(w2v_model.wv.vocab)))
        print('Limit: ' + str(cluster_limit))   
        while clusters > cluster_limit:
            
            if clusters % 100 == 0:
                print("clusters: ", clusters)
            
            # Find the closest pair
            no_neighb = cluster.find_closest_pair(cluster_name)
            if no_neighb:
                for potential_nb in cluster.pairs.keys():
                    if test:
                        #print('Here!')
                        test = False
                    if cluster_name is not potential_nb:
                        dist = RHP.cosine_dist(cluster.pairs[cluster_name]['center'], cluster.pairs[potential_nb]['center'])
                        cluster.distances_pairs = cluster.distances_pairs.append({'Word_1': cluster_name, 'Word_2': potential_nb, 'Distance': dist}, ignore_index=True)
                        ##print("Distance = ", cluster.distances_pairs.append({'Word_1': cluster_name, 'Word_2': potential_nb, 'Distance': dist}, ignore_index=True))
                        #print("----------")
                #print("Before sort: \n", cluster.distances_pairs.head())
                cluster.distances_pairs = cluster.distances_pairs.sort_values(by=['Distance'])
                #print("After sort: \n", cluster.distances_pairs.head())

            #if no_neighb:
                #cluster_name = None
                #continue
            # Cluster the closest pair and update the vocab
            cluster_name, word_emb_1_to_delete, word_emb_2_to_delete = cluster.cluster_closest()
            word_id+=1
            # Hash the word embedding and add to index
            w2v_index.index(cluster.pairs[cluster_name]['center'], word_id)
            # Add the new cluster to the id dict for extraction from hash
            w2v_model.wv.index2word.append(cluster_name)
            clusters-=1

            # Delete the merged words from the index
            w2v_index.delete_index(word_emb_1_to_delete)
            w2v_index.delete_index(word_emb_2_to_delete)

        cluster.save_pairs(path)

    
    
    def init_clustering(self, w2v_model, num_clusters, path):
        '''
        if not os.path.exists(path):
            cluster_vocab(w2v_model, num_clusters, path)
        '''
        self.cluster_vocab(w2v_model, num_clusters, path)

        with open(path, 'r') as myfile:
            clustered_data = json.load(myfile)

        return clustered_data
    
    def topic_factor(self, dt_sub_matrix):
        ones = np.ones(dt_sub_matrix.shape[0], dtype=int)
        ones2 = np.ones(dt_sub_matrix.shape[1], dtype=int)
        f = 1 / (ones.T @ dt_sub_matrix @ ones2) * (dt_sub_matrix.T @ ones)
        d = np.dot(ones.T @ dt_sub_matrix, f) / np.dot(f, f.T)

        return f, d
    
    def topic_modeling(self):
        # Algorithm 3 and pre-requisites
        clusters = []    
        cluster_words = []
        # if no clusters specified, run on all clusters. otherwise run on those specified
        for (k,v) in self.clusters.items():
            clusters.append(k)
            cluster_words.append(v['words'])


        # Find the frequency counts for the correct documents
        fds = np.array(self.training_data['freq_counts'])

        # TODO: Why do this?
        texts = self.training_data['prepped_total_text_cleaned']
        dicts = [dict(i) for i in fds]

        # Algo 3
        np.seterr(all = 'ignore') #ignorerar varningar för div med 0 samt nan --- måste nog hantera detta
        fs,ds = [],[]
        for ci,c in enumerate(clusters):#number of clusters
            arr = np.zeros((len(texts),len(cluster_words[ci])),dtype=int)
            for i,word in enumerate(cluster_words[ci]):#number of words in the cluster
                for j,doc in enumerate(dicts):#number of documents
                    if word in dicts[j]:
                        arr[j][i] = dicts[j][word]
            f,d = self.topic_factor(arr)
            fs.append(f)
            ds.append(d)
            #print(arr)

        #print(fs)
        #print(ds)

        assert len(clusters) == len(fs) == len(ds)

        # Create triplets containing Si, Fi and di representing each textual factor
        textual_factors = []
        for i in range(len(clusters)):
            words = cluster_words[i]
            Fi = fs[i]
            di = ds[i]
            triplet = (words, Fi, di)
            textual_factors.append(triplet)

        return textual_factors

    
    def add_beta_loading(self, data, beta_loader):    
        betas = []
        for _, row in data.iterrows():
            # Save the beta-loading in the dataframe
            beta = beta_loader.beta_loading(row['freq_counts'])
            betas.append(beta)

        data_with_beta = data.assign(beta_loading = betas)
        return data_with_beta
    
    def create_classifier_and_pca(self, x_train, y_train, onehot=False):
        from sklearn.ensemble import RandomForestClassifier

        print('Creating model and PCA....')
        if onehot:
            #x_train = self.PCA.transform(x_train)
            print('training data transormed')
            self.SVM = svm.SVC(kernel='linear')
            print('linear training started')
            self.SVM.fit(x_train, y_train)
            print('linear training ended and radial started')
            self.SVM_radial = svm.SVC()
            self.SVM_radial.fit(x_train, y_train)
            print('radial ended and RF started')

            self.RF = RandomForestClassifier(n_estimators = 500)
            self.RF.fit(x_train, y_train)
            print('RD ended')
        else:
            self.PCA = PCA(n_components='mle')
            self.PCA.fit(x_train)
            print('PCA created')
            x_train = self.PCA.transform(x_train)
            print('training data transormed')
            self.SVM = svm.SVC(kernel='linear')
            print('linear training started')
            self.SVM.fit(x_train, y_train)
            print('linear training ended and radial started')
            self.SVM_radial = svm.SVC()
            self.SVM_radial.fit(x_train, y_train)
            print('radial ended and RF started')

            self.RF = RandomForestClassifier(n_estimators = 500)
            self.RF.fit(x_train, y_train)
            print('RD ended')

        
        
    def runCombined(self):
        w2v = Word2Vec_routine(self.training_data['prepped_total_text_cleaned'])
        w2v.create_sentence_vec_represenation()
        w2v.create_embedding()
        print(len(self.training_data['prepped_total_text_cleaned']))
        
        data_len = len(self.training_data['prepped_total_text_cleaned'])
        
        x_train = np.zeros((data_len, 300))
        for idx, doc in enumerate(w2v._article_vector_representation):
            word_vecs = [w2v.w2v_model.wv[word] for word in doc if word in w2v.w2v_model.wv.vocab]
            if len(word_vecs) < 10:
                print(len(word_vecs))
            if len(word_vecs) == 0:
                print("empty")
            else:
                x_train[idx] = np.mean(word_vecs, axis=0)
        print(x_train.shape)
        y_train = self.training_data['recommendationId']
        self.create_classifier_and_pca(x_train, y_train, False)

        
        w2v_test = Word2Vec_routine(self.testing_data['prepped_total_text_cleaned'])
        w2v_test.create_sentence_vec_represenation()
        w2v_test.create_embedding()
        data_len_train = len(self.testing_data['prepped_total_text_cleaned'])
        
        x_test = np.zeros((data_len_train, 300))
        print(x_test.shape)
        for idx, test_doc in enumerate(w2v_test._article_vector_representation):
            word_vecs = [w2v_test.w2v_model.wv[word] for word in test_doc if word in w2v_test.w2v_model.wv.vocab]
            if len(word_vecs) < 10:
                print(len(word_vecs))
            if len(word_vecs) == 0:
                print("empty")
            else:
                x_test[idx] = np.mean(word_vecs, axis=0)
        y_test = self.testing_data['recommendationId']
        
        print(x_test.shape)
        print(len(x_test))
        print(len(y_test))
        print('Classifier created')
        print('PCA')
        x_test = self.PCA.transform(x_test)
        print('SVM')
        score_svm = self.SVM.score(x_test, y_test)
        print('radial svm')
        score_radial_svm = self.SVM_radial.score(x_test,y_test)
        print('Random Forest')
        score_rf = self.RF.score(x_test,y_test)
        print('should return scores now')
        return {
            'score_svm': score_svm,
            'score_radial_svm': score_radial_svm,
            'score_rf': score_rf,
            'importance': list(self.RF.feature_importances_)
        }
            
            
    def generate_one_hot(self, vocab, sentence_representation, data):
        features = np.zeros((len(sentence_representation), len(vocab)))
        for doc_id, document in enumerate(sentence_representation):
            one_hot = np.zeros((len(vocab)))
            for idx, term in enumerate(vocab):
                if term in document:
                    one_hot[idx] = 1
            features[doc_id] = one_hot
        return features, data['recommendationId']
                    
                
    
    def run_one_hot(self, vocab, sentence_representation):
        print('One hot started')
                
        w2v = Word2Vec_routine(self.training_data['prepped_total_text_cleaned'])
        w2v.create_sentence_vec_represenation()
        print('Sentence article presentation: ', len(w2v._article_vector_representation))
        x_train, y_train = self.generate_one_hot(vocab, w2v._article_vector_representation, self.training_data)

        w2v_test = Word2Vec_routine(self.testing_data['prepped_total_text_cleaned'])
        w2v_test.create_sentence_vec_represenation()
        print('Sentence article presentation: ', len(w2v_test._article_vector_representation))
        x_test, y_test = self.generate_one_hot(vocab, w2v_test._article_vector_representation, self.testing_data)
        
        print('one-hot encoding done, creating classifier....')
        self.create_classifier_and_pca(x_train, y_train, True)
        print(x_test)
        print(len(x_test))
        print(len(y_test))
        print('Classifier created')
        print('PCA')
        #x_test = self.PCA.transform(x_test)
        print('SVM')
        score_svm = self.SVM.score(x_test, y_test)
        print('radial svm')
        score_radial_svm = self.SVM_radial.score(x_test,y_test)
        print('Random Forest')
        score_rf = self.RF.score(x_test,y_test)
        print('should return scores now')
        return {
            'score_svm': score_svm,
            'score_radial_svm': score_radial_svm,
            'score_rf': score_rf,
            'importance': list(self.RF.feature_importances_)
        }
    
    def run_experiment(self):
        # Create a vector embedding for the words of the vocab
        print("Create w2v model")
        self.w2v_model = self.init_w2v_model(self.w2v_path, self.training_data, self.vec_size, self.window, self.min_count, self.negative, self.alpha, self.min_alpha, self.sample, self.google)
        # Cluster the words with the help of lsh
        print("Cluster the words")
        self.clusters = self.init_clustering(self.w2v_model, self.num_clusters, self.cluster_path)

        # Topic modeling for retrieving the textual factors
        print("Topic modeling")
        textual_factors = self.topic_modeling()

        # Initialize the Beta Loader with the textual factors
        beta_loader = Beta_Loading(textual_factors)

        # Add beta-loadings to the dataframes for both training and test-data
        print("Transform documents into beta_loadings")
        df_train = self.add_beta_loading(self.training_data, beta_loader)
        df_test = self.add_beta_loading(self.testing_data, beta_loader)


        # Initialize classifier
        print("Create Classifier")

        x_train, y_train = tuple(df_train['beta_loading'].values), tuple(df_train['recommendationId'].values)

        # Create a classifier and pca for both scenarios
        self.create_classifier_and_pca(x_train, y_train)

        # Get score on whole dataset with hold
        x_test = tuple(df_test['beta_loading'].values)
        y_test = tuple(df_test['recommendationId'].values)

        if self.google:
            print("Google results:")
        x_test = self.PCA.transform(x_test)
        full_score = self.SVM.score(x_test, y_test)
        print("Score with Hold included lineaer svm: ", full_score)
        
        radial_score = self.SVM_radial.score(x_test, y_test)
        print("Score with Hold included radial svm: ", radial_score)
        
        full_score_rf_own = self.RF.score(x_test, y_test)
        print("RF: Score with Hold included: ", full_score_rf_own)    

        # Create dict of the hyper-params
        keys = ['vec_size', 'n_bit', 'n_table', 'num_clusters', 'window', 'min_count', 'negative', 'alpha', 'min_alpha', 'sample']
        values = [self.vec_size, self.n_bit, self.n_table, self.num_clusters, self.window, self.min_count, self.negative, self.alpha, self.min_alpha, self.sample]
        hyper_params = dict(zip(keys, values))


        return {
            'hyper-params': hyper_params,
            'full_score': full_score,
            'full_score_rf': full_score_rf_own,
            'radial_score': radial_score,
            'importance': list(self.RF.feature_importances_)
        }
