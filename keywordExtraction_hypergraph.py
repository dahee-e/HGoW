import argparse
import os
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from collections import Counter
import eval_set
import index
import HGoW_generate
from gowpy.gow.builder import GoWBuilder

from gowpy.summarization.unsupervised import CoreRankKeywordExtractor
from keybert import KeyBERT

if __name__ == "__main__":
    n = 5
    file_path = './datasets/semeval/'
    file_name = 'C-1.stc'
    sentences = HGoW_generate.load_set_from_file(file_path+file_name)
    text = HGoW_generate.text_preprocessing(sentences)
    G,E = HGoW_generate.load_hypergraph(text)
    T = index.naive_index_construction(G, E)
    keywords =sum(T.leaf_count(),[])
    kgcore = Counter(keywords).most_common()
    kgcore_keyword = [x[0] for x in kgcore]
    '''hypergraph'''
    print("hypergraph:", kgcore_keyword)

    '''coreness'''
    text = [str(token) for x in text for token in x ]
    extractor_kw_cr = CoreRankKeywordExtractor(directed=False, weighted=True, window_size=8)
    coreness = extractor_kw_cr.extract(' '.join(text))
    coreness_keyword = [x[0] for x in coreness]
    print("coreness:",coreness_keyword)



    '''centrality'''
    builder = GoWBuilder(directed=False, window_size=8)
    sentence_not_token = HGoW_generate.text_preprocessing_not_token(sentences)
    G1 = builder.compute_gow_from_document(sentence_not_token)
    g = G1.to_labeled_graph()

    # print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True))
    degree_centrality = nx.degree_centrality(g)
    degree_cen_keyword = [x[0] for x in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:n]]
    print("degree_centrality:",degree_cen_keyword)

    eigenvector_centrality = nx.eigenvector_centrality(g)
    eigenvector_cen_keyword = [x[0] for x in sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:n]]
    print("eignevector_centrality:",eigenvector_cen_keyword)

    '''keyBERT'''
    keybert = KeyBERT().extract_keywords(sentence_not_token)
    keybert_keyword = [x[0] for x in keybert[:n]]
    print("keyBERT:",keybert_keyword)
    '''evaluation'''
    kgcore_eval, answer_eval = eval_set.evaluation_pre(kgcore_keyword, file_path+file_name.split('.')[0]+'.key')
    print("kgcore_f1:",f1_score(kgcore_eval, answer_eval))
    print("kgcore_precision:",precision_score(kgcore_eval, answer_eval))
    print("kgcore_recall:",recall_score(kgcore_eval, answer_eval))

    coreness_eval, answer_eval = eval_set.evaluation_pre(coreness_keyword, file_path + file_name.split('.')[0] + '.key')
    print("coreness_f1:", f1_score(coreness_eval, answer_eval))
    print("coreness_precision:", precision_score(coreness_eval, answer_eval))
    print("coreness_recall:", recall_score(coreness_eval, answer_eval))

    degree_cen_eval, answer_eval = eval_set.evaluation_pre(degree_cen_keyword, file_path + file_name.split('.')[0] + '.key')
    print("degree_cen_f1:", f1_score(degree_cen_eval, answer_eval))
    print("degree_cen_precision:", precision_score(degree_cen_eval, answer_eval))
    print("degree_cen_recall:", recall_score(degree_cen_eval, answer_eval))

    eigenvector_cen_eval, answer_eval = eval_set.evaluation_pre(eigenvector_cen_keyword, file_path + file_name.split('.')[0] + '.key')
    print("eigenvector_cen_f1:", f1_score(eigenvector_cen_eval, answer_eval))
    print("eigenvector_cen_precision:", precision_score(eigenvector_cen_eval, answer_eval))
    print("eigenvector_cen_recall:", recall_score(eigenvector_cen_eval, answer_eval))

    keybert_eval, answer_eval = eval_set.evaluation_pre(keybert_keyword, file_path + file_name.split('.')[0] + '.key')
    print("keybert_f1:", f1_score(keybert_eval, answer_eval))
    print("keybert_precision:", precision_score(keybert_eval, answer_eval))
    print("keybert_recall:", recall_score(keybert_eval, answer_eval))