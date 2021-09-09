'''
This implementation was inpired by https://github.com/dongguosheng/lsh
'''

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
from experiment import Experiment




from w2v import Word2Vec_routine
from rhp import RHP
from index import Index
from clustering import Cluster
from beta_loading import Beta_Loading, mock_data_beta_loading


# --------------------- Data --------------------- 
def partition_data(data, test_size, train_size, seed):
    # Partition the data into training- and test-sets
    # test_size can either be a fraction, or an actual number of samples to use 
    train, test = train_test_split(data, test_size=test_size, train_size=train_size, random_state=seed)
    return train, test


def split_by_label(data):
    df_buy = data[data.recommendationId == 1]
    df_hold = data[data.recommendationId == 2]
    df_sell = data[data.recommendationId == 3]
    return df_buy, df_sell, df_hold


def get_balanced_dataset(df, drop_hold = False):
    #Most uncommon
    class_limit = min(df['recommendationId'].value_counts())
    balanced_df = df[df.recommendationId == 1][:class_limit]
    if not drop_hold:
        balanced_df = pandas.concat([df[df.recommendationId == 2][:class_limit], balanced_df])
    balanced_df = pandas.concat([df[df.recommendationId == 3][:class_limit], balanced_df])
    return tuple(balanced_df['beta_loading'].values), tuple(balanced_df['recommendationId'].values)


def get_balanced_dataset_as_full_frame(df, drop_hold = False):
    #Most uncommon
    class_limit = min(df['recommendationId'].value_counts())
    balanced_df = df[df.recommendationId == 1][:class_limit]
    if not drop_hold:
        balanced_df = pandas.concat([df[df.recommendationId == 2][:class_limit], balanced_df])
    balanced_df = pandas.concat([df[df.recommendationId == 3][:class_limit], balanced_df])
    return balanced_df

# --------------------- Word2Vec --------------------- 
    
def experiment(google, normal):
    from matplotlib import pyplot
    X = google[google.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(normal.wv.vocab)
    for i, word in enumerate(words[:100]):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.savefig('google_plt')
    
    pyplot.clf()
    X = normal[normal.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(normal.wv.vocab)
    for i, word in enumerate(words[:100]):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.savefig('normal_plt')

def create_w2v_model(path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google):
    # Create a w2v with the given training data

    word2vec = Word2Vec_routine(train_data['prepped_total_text_cleaned'])
    word2vec.create_sentence_vec_represenation()
    word2vec.create_embedding(vec_size, window, min_count, negative, alpha, min_alpha, sample, google)
    word2vec.save_model(path)
    print("Starting intersection...")
    #w2v_google = word2vec.getGoogle()
    #w2v_google = word2vec.w2v_model.intersect_word2vec_format('./GoogleNews-vectors-negative300.bin', lockf=1.0,binary=True)
    #w2v_google = word2vec.create_google_w2v_model()
    #w2v_google.save('w2v_google')
    
    #experiment(w2v_google, word2vec.w2v_model)
    
    
    #word2vec.w2v_model.intersect_word2vec_format(google_w2v, lockf=1.0,binary=True)
    return word2vec.w2v_model

def init_w2v_model(path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google):
    '''
    if os.path.exists(path):
        w2v = models.Word2Vec.load(path)
    else:
        w2v = create_w2v_model(path, train_data)
    '''
    w2v = create_w2v_model(path, train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google)
    return w2v


# --------------------- Clustering --------------------- 
def cluster_vocab(w2v_model, target_num_clusters, path):    
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
        '''
        if clusters % 100 == 0:
            print("clusters: ", clusters)
        '''
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

def init_clustering(w2v_model, num_clusters, path):
    '''
    if not os.path.exists(path):
        cluster_vocab(w2v_model, num_clusters, path)
    '''
    cluster_vocab(w2v_model, num_clusters, path)
    
    with open(path, 'r') as myfile:
        clustered_data = json.load(myfile)

    return clustered_data


# --------------------- Topic Modeling --------------------- 
def topic_factor(dt_sub_matrix):
    ones = np.ones(dt_sub_matrix.shape[0], dtype=int)
    ones2 = np.ones(dt_sub_matrix.shape[1], dtype=int)
    f = 1 / (ones.T @ dt_sub_matrix @ ones2) * (dt_sub_matrix.T @ ones)
    d = np.dot(ones.T @ dt_sub_matrix, f) / np.dot(f, f.T)

    return f, d

def topic_modeling(train_data, clustered_data):
    # Algorithm 3 and pre-requisites
    clusters = []    
    cluster_words = []
    # if no clusters specified, run on all clusters. otherwise run on those specified
    for (k,v) in clustered_data.items():
        clusters.append(k)
        cluster_words.append(v['words'])

    # Extract the ids for the training data
    # idx = [idx for idx in train_data['idx']]

    # Find the frequency counts for the correct documents
    # fds = np.array(pandas.read_pickle('freq_dists.pickle'))[idx]
    fds = np.array(train_data['freq_counts'])

    # TODO: Why do this?
    texts = train_data['prepped_total_text_cleaned']
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
        f,d = topic_factor(arr)
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


# --------------------- Beta Loading --------------------- 
def add_beta_loading(data, beta_loader):    
    data['beta_loading'] = np.nan
    data['beta_loading'] = data['beta_loading'].astype('object')
    for i, row in data.iterrows():
        # Save the beta-loading in the dataframe
        beta = beta_loader.beta_loading(row['freq_counts'])
        data.at[i, 'beta_loading'] = beta

    return data


# --------------------- Classifier --------------------- 
def create_classifier_and_pca(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    print('X_train; ', len(x_train))
    pca = PCA(n_components='mle')
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    classifier = svm.SVC()
    classifier.fit(x_train, y_train)
    
    rf_clf = RandomForestClassifier(n_estimators = 3000)
    rf_clf.fit(x_train, y_train)
    
    
    return classifier, pca, rf_clf


def predict(test, classifier, pca):
    x_test = tuple(test['beta_loading'].values)
    x_test = pca.transform(x_test)
    y_test = tuple(test['recommendationId'].values)
    score = classifier.score(x_test, y_test)
    return score


def run_experiment(df_train_data, df_test_data, vec_size, n_bit, n_table, num_clusters, w2v_path, cluster_path, window, min_count, negative, alpha, min_alpha, sample, google = False):

    # Create a vector embedding for the words of the vocab
    print("Create w2v model")
    w2v_model = init_w2v_model(w2v_path, df_train_data, vec_size, window, min_count, negative, alpha, min_alpha, sample, google)
    # Cluster the words with the help of lsh
    print("Cluster the words")
    clusters = init_clustering(w2v_model, num_clusters, cluster_path)
    #cluster_google = init_clustering(w2v_google, num_clusters, 'clusters_google.json')

    # Topic modeling for retrieving the textual factors
    print("Topic modeling")
    textual_factors = topic_modeling(df_train_data, clusters)
    #textual_factors_google = topic_modeling(df_train_data, cluster_google)

    # Initialize the Beta Loader with the textual factors
    beta_loader = Beta_Loading(textual_factors)
    #beta_loader_google = Beta_Loading(textual_factors_google)
    

    # Add beta-loadings to the dataframes for both training and test-data
    print("Transform documents into beta_loadings")
    df_train = add_beta_loading(df_train_data, beta_loader)
    df_test = add_beta_loading(df_test_data, beta_loader)
    
    
    #df_train_google = add_beta_loading(df_train_data_google, beta_loader_google)
    #df_test_google = add_beta_loading(df_test_data_google, beta_loader_google)

    
    # Initialize classifier
    print("Create Classifier")
    
    x_train, y_train = tuple(df_train['beta_loading'].values), tuple(df_train['recommendationId'].values)
    #x_train_google, y_train_google = tuple(df_train_google['beta_loading'].values), tuple(df_train_google['recommendationId'].values)

    # x_train, y_train = get_balanced_dataset(df_train)
    # x_train_no_hold, y_train_no_hold = get_balanced_dataset(df_train, True)
    
    # x_train_google, y_train_google = get_balanced_dataset(df_train_google)
    # x_train_no_hold_google, y_train_no_hold_google = get_balanced_dataset(df_train_google, True)
    #x_train = tuple(df_train['beta_loading'].values)
    #y_train = tuple(df_train['recommendationId'].values)

    
    # Create a classifier and pca for both scenarios
    classifier, pca, rf_own = create_classifier_and_pca(x_train, y_train)
    #classifier_no_hold, pca_no_hold = create_classifier_and_pca(x_train_no_hold, y_train_no_hold)
    
    #classifier_google, pca_google, rf_google = create_classifier_and_pca(x_train_google, y_train_google)
    #classifier_no_hold_google, pca_no_hold_google = create_classifier_and_pca(x_train_no_hold_google, y_train_no_hold_google)


    # Extract test-sets for each label
    #test_buy, test_sell, test_hold = split_by_label(df_test)
    
    
    # Get scores by label
    #for classifier_source, pca_source, df_test_source in [zip(classifier, pca, df_test), zip(classifier_google, pca_google, df_test_google)]:
        #test_buy, test_sell, test_hold = split_by_label(df_test_source)
        #buy_score = predict(test_buy, classifier_source, pca_source)
        #sell_score = predict(test_sell, classifier_source, pca_source)
        #hold_score = predict(test_hold, classifier_source, pca_source)
        #print("Buy: " + str(buy_score))
        #print("Sell: " + str(sell_score))
        #print("Hold: " + str(hold_score))

    # Get score on whole dataset with hold
    x_test = tuple(df_test['beta_loading'].values)
    y_test = tuple(df_test['recommendationId'].values)
    
    #x_test_google = tuple(df_test_google['beta_loading'].values)
    #y_test_google = tuple(df_test_google['recommendationId'].values)
    if google:
        print("Google results:")
    x_test = pca.transform(x_test)
    full_score = classifier.score(x_test, y_test)
    print("Score with Hold included: ", full_score)
    
    #x_test_google = pca_google.transform(x_test_google)
    #full_score_google = classifier_google.score(x_test_google, y_test_google)
    #print("Score with Hold included for google: ", full_score_google)
    
    full_score_rf_own = rf_own.score(x_test, y_test)
    print("RF: Score with Hold included: ", full_score_rf_own)    
    print("Importances: ", list(rf_own.feature_importances_))
    
    #full_score_rf_google = rf_google.score(x_test_google, y_test_google)
    #print("RF: Score with Hold included for google: ", full_score_rf_google)
    #print("Importances own: ", list(rf_google.feature_importances_))
    
    
    # Get score on whole dataset without hold

    #x_test_no_hold = tuple(pandas.concat([test_buy, test_sell])['beta_loading'].values)
    #y_test_no_hold = tuple(pandas.concat([test_buy, test_sell])['recommendationId'].values)
    
    
    #x_test_no_hold = pca_no_hold.transform(x_test_no_hold)
    #print(len(x_test_no_hold[0]))
    #full_score_no_hold = classifier_no_hold.score(x_test_no_hold, y_test_no_hold)
    #print("Score with Hold excluded", full_score_no_hold)
    
    # Create dict of the hyper-params
    keys = ['vec_size', 'n_bit', 'n_table', 'num_clusters', 'window', 'min_count', 'negative', 'alpha', 'min_alpha', 'sample']
    values = [vec_size, n_bit, n_table, num_clusters, window, min_count, negative, alpha, min_alpha, sample]
    hyper_params = dict(zip(keys, values))

    
    return {
        'hyper-params': hyper_params,
        'full_score': full_score,
        'full_score_rf': full_score_rf_own,
        'importance': list(rf_own.feature_importances_)
    }


if __name__ == "__main__":
    # Measure time
    dateTimeObj = datetime.now()
    print(dateTimeObj)
    
    # Set this depending on if you want a fast, small run, or an actual run with the full dataset
    full_run = True

    if full_run:
        # For actual proper runs
        test_size = 0.20
        train_size = None
        w2v_path = 'w2v.model_full'
        num_clusters = 80 # No longer used
        cluster_path = 'clusters.json'

    else:
        # For testing purposes
        test_size = 0.20
        train_size = 5000
        w2v_path = 'w2v.model_short'
        num_clusters = 10 # No longer used
        cluster_path = 'clusters-short.json'
        
    # Params w2v:
    windows = [5]
    min_counts = [5]
    negatives = [0.1]
    alphas = [0.8]
    min_alphas = [0.001]
    samples = [1e-1]
    k_fold = 4
    
    # Params
    vector_sizes = [300]
    n_bits = [16]
    n_tables = [5]
    cluster_limits = [300]
    
    # Setup the data
    df_all = pandas.read_pickle('processed_data_fin.pickle')
    
    # Add a column for the freq-counts
    df_all['freq_counts'] = pandas.read_pickle('freq_dists.pickle')

    # Remove unlabeled data
    df_all = df_all[df_all.recommendationId != 0] 
    
    # Split the data into training and test sets
    # Notice the seed, used for testing purposes
    print("Partiton data")

    # Shuffle and reset indexes
    #df_all = shuffle(df_all, random_state=321)
    #df_all.reset_index(inplace=True, drop=True)

    # Retrieve a balanced dataset
    df_balanced = get_balanced_dataset_as_full_frame(df_all)
    # Partition data into training and test
    df_train, df_test = partition_data(df_balanced, test_size, train_size, seed=321)
    
    
    n = len(df_balanced)//k_fold  #chunk row size
    data_chunks = [df_balanced[i:i+n] for i in range(0, len(df_balanced) ,n)]
    # Check if the last chunk is significantly smaller than the others, and if so discard it
    if len(data_chunks[0]) * 0.9 > len(data_chunks[-1]):
        data_chunks = data_chunks[:-1]
    
    # Create multiple different runs with different hyper-parameters
    runs = [[vec_size, n_bit, n_table, cluster_lim, window, min_count, negative, alpha, min_alpha, sample, k_sample] 
                                                    for vec_size in vector_sizes 
                                                    for n_bit in n_bits 
                                                    for n_table in n_tables
                                                    for cluster_lim in cluster_limits
                                                    for window in windows
                                                    for min_count in min_counts
                                                    for negative in negatives
                                                    for alpha in alphas
                                                    for min_alpha in min_alphas
                                                    for sample in samples
                                                    for k_sample in range(k_fold)]
    count = 0
    res_dir = 'res/'
    
    #for [vec_size, n_bit, n_table, cluster_lim, window, min_count, negative, alpha, min_alpha, sample, k_sample] in runs:
        #df_test = data_chunks[k_sample]
        #df_train = pandas.DataFrame()
        #print("DF_test = ", df_test)
    #for i in range(k_fold):
        #    if i != k_sample:
        #        df_train = pandas.concat([df_train, data_chunks[i]])
        
    from sklearn.model_selection import KFold

    importance_own = None
    importance_google = None
    importance_one_hot = None
    kf = KFold(n_splits=10, shuffle=True)
    
    scores_svm_own = []
    scores_rf_own = []
    scores_radial_own = []
    
    scores_svm_google = []
    scores_rf_google = []
    scores_radial_google = []
    
    scores_one_hot_linear = []
    scores_one_hot_radial = []
    scores_one_hot_rf = []
    
    
    for train_index, test_index in kf.split(df_balanced):
        x_train, x_test = df_balanced.iloc[train_index], df_balanced.iloc[test_index]
    
        from operator import add
        
        #combined_wordvec_experiment = Experiment(x_train, x_test, vec_size=300, n_bit=16, n_table=5, num_clusters=300, w2v_path=w2v_path, cluster_path=cluster_path, window=15, min_count=30, negative=0.1, alpha=0.05, min_alpha=0.001, sample=1e-1, google = False )
        
        #experiment_combined_word_vector = combined_wordvec_experiment.runCombined()
        #print(experiment_combined_word_vector)
        print('expirement own init')
        experiment_own_object = Experiment(x_train, x_test, vec_size=300, n_bit=16, n_table=5, num_clusters=300, w2v_path=w2v_path, cluster_path=cluster_path, window=15, min_count=30, negative=0.1, alpha=0.05, min_alpha=0.001, sample=1e-1, google = False )
        print('expirement own started')
        experiment_own = experiment_own_object.run_experiment()
        print('expirement own done')
        if importance_own is None:
            importance_own = experiment_own['importance']
        else:
            importance_own = list(map(add, importance_own, experiment_own['importance']))
            
        scores_svm_own.append(experiment_own['full_score'])
        scores_rf_own.append(experiment_own['full_score_rf'])
        scores_radial_own.append(experiment_own['radial_score'])
        
        print('svm own: ', experiment_own['full_score'])
        print('RF own; ', experiment_own['full_score_rf'])
        

        
        print('init google')
        experiment_google_object = Experiment(x_train, x_test, vec_size=300, n_bit=16, n_table=5, num_clusters=300, w2v_path=w2v_path, cluster_path='clusters_google.json', window=15, min_count=30, negative=0.1, alpha=0.05, min_alpha=0.001, sample=1e-1, google = True )
        print('Experiment google')
        experiment_google = experiment_google_object.run_experiment()
        print('Google experiment done')
        if importance_google is None:
            importance_google = experiment_google['importance']
        else:
            importance_google = list(map(add, importance_google, experiment_google['importance']))

        scores_svm_google.append(experiment_google['full_score'])
        scores_rf_google.append(experiment_google['full_score_rf'])
        scores_radial_google.append(experiment_google['radial_score'])
        
        print('google svm score: ', experiment_google['full_score'])
        print('google rf score; ', experiment_google['full_score_rf'])
        
        
        w2v_own = experiment_own_object.w2v_model
        sentences = experiment_own_object.w2v_sentences
        
        exp_one_hot = Experiment(x_train, x_test, vec_size=300, n_bit=16, n_table=5, num_clusters=300, w2v_path=w2v_path, cluster_path=cluster_path, window=15, min_count=30, negative=0.1, alpha=0.05, min_alpha=0.001, sample=1e-1, google = False)
    
        one_hot_own = exp_one_hot.run_one_hot(w2v_own.wv.vocab, sentences)
        
        scores_one_hot_linear.append(one_hot_own['score_svm'])
        scores_one_hot_radial.append(one_hot_own['score_radial_svm'])
        scores_one_hot_rf.append(one_hot_own['score_rf'])
        
        if importance_one_hot is None:
            importance_one_hot = one_hot_own['importance']
        else:
            importance_one_hot = list(map(add, importance_one_hot, one_hot_own['importance']))
        print('One hot results: ')
        print(one_hot_own)
        
        
        break
        
        
    #res_string = res_dir + 'res_w2v_final_with_google' + str(count) + '.json'
    #with open(res_string, 'w') as write_file:
        #json.dump(res, write_file, indent=4, sort_keys=True)

    print('Mean score of own model with SMV: ', np.mean(scores_svm_own), ' and RF: ', np.mean(scores_rf_own), ' and radial: ', np.mean(scores_radial_own))
    print("-------")
    print('Individual scores for each run of own model SVM: ', scores_svm_own, ' and RF: ', scores_rf_own, ' and radial: ', scores_radial_own)
    print("-------")
    print('Mean score of google model with SMV: ', np.mean(scores_svm_google), ' and RF: ', np.mean(scores_rf_google), ' and radial: ', np.mean(scores_radial_google))
    print("-------")
    print('Individual scores for each run of google model SVM: ', scores_svm_google, ' and RF: ', scores_rf_google, ' and radial: ', scores_radial_google)
    print("-------")
    print('Mean score of one hot model, linear: ', np.mean(scores_one_hot_linear), ' and RF: ', np.mean(scores_one_hot_rf), ' and radial: ', np.mean(scores_one_hot_radial))
    print("-------")
    print('Individual scores per test run one hot, linear: ', scores_one_hot_linear, ' and RF: ', scores_one_hot_rf, ' and radial: ', scores_one_hot_radial)
    print("-------")    
    
    
    from matplotlib import pyplot
    
    plt = pyplot.bar([t for t in range(len(importance_own))], importance_own)
    pyplot.savefig('importnace_own')
    
    pyplot.clf()
    plt = pyplot.bar([t for t in range(len(importance_google))], importance_google)
    pyplot.savefig('importnace_google')
    
    pyplot.clf()
    plt = pyplot.bar([t for t in range(len(importance_google))], importance_google)
    pyplot.savefig('importnace_one_hot')
    
    
    for one_hot_linear, own_linear, google_linear in zip(scores_one_hot_linear, scores_svm_own, scores_svm_google):
        print('linear scores')
        print('own/one = ', own_linear/one_hot_linear)
        print('google/one= ', google_linear/one_hot_linear)
        
    for one_hot_radial, own_radial, google_radial in zip(scores_one_hot_radial, scores_radial_own, scores_radial_google):
        print('Radial scores')
        print('own/one= ', own_radial/one_hot_radial)
        print('google/one= ', google_radial/one_hot_radial)
        
    for one_got_rf, own_rf, google_rf in zip(scores_one_hot_rf, scores_rf_own, scores_rf_google):
        print('RF scores')
        print('own/one= ', own_rf/one_got_rf)
        print('google/one= ', google_rf/one_got_rf)
    
    
        scores = {
            'own_linear': np.mean(scores_svm_own),
            'google_linear': np.mean(scores_svm_google),
            'one_hot_linear':np.mean(scores_one_hot_linear),
            'own_radial': np.mean(scores_radial_own),
            'google_radial': np.mean(scores_radial_google),
            'one_hot_radial': np.mean(scores_one_hot_radial),
            'own_RF': np.mean(scores_rf_own),
            'google_rf': np.mean(scores_rf_google),
            'one_hot_rf': np.mean(scores_one_hot_rf),
            'folds_own_linear': scores_svm_own,
            'folds_google_linear': scores_svm_google,
            'folds_one_hot_linear': scores_one_hot_linear,
            'folds_own_radial': scores_radial_own,
            'folds_google_radial': scores_radial_google,
            'folds_one_hot_radial': scores_one_hot_radial,
            'folds_own_rf': scores_rf_own,
            'folds_google_rf': scores_rf_google,
            'folds_one_hot_rf': scores_one_hot_rf
            
        }
        
        with open('computed_results.json', 'w') as write_file:
            json.dump(scores, write_file, indent=4, sort_keys=True)
        print("dumped")
    
    
    
    dateTimeObj = datetime.now()
    print(dateTimeObj)
    
    
    