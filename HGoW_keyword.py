import argparse
import os
import networkx as nx
import pandas as pd
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
    #file_path = './datasets/Hulth2003/validation_training/validation/'
    #file_path = './datasets/Marujo2012/train_test/train/'
    file_path = './datasets/semeval/'
    kgcore_f1 = 0
    coreness_f1 = 0
    degree_cen_f1 = 0
    eigenvector_cen_f1 = 0
    keybert_f1 = 0

    kgcore_precision = 0
    coreness_precision = 0
    degree_cen_precision = 0
    eigenvector_cen_precision = 0
    keybert_precision = 0

    kgcore_recall = 0
    coreness_recall = 0
    degree_cen_recall = 0
    eigenvector_cen_recall = 0
    keybert_recall = 0
    size = 0
    for file_name in os.listdir(file_path):
        if file_name.endswith(".stc"):
            size += 1
            print(file_path+file_name)
            sentences = HGoW_generate.load_set_from_file(file_path+file_name)
            text = HGoW_generate.text_preprocessing(sentences)

            n = 5
            '''hypergraph'''
            G,E = HGoW_generate.load_hypergraph(text)
            T = index.naive_index_construction(G, E)
            keywords =sum(T.leaf_count(),[])
            kgcore = Counter(keywords).most_common()
            kgcore_keyword = [x[0] for x in kgcore]
            if len(kgcore_keyword) > n:
                kgcore_keyword = kgcore_keyword[:n]
            # n = len(kgcore_keyword)


            '''coreness'''
            text = [str(token) for x in text for token in x ]
            extractor_kw_cr = CoreRankKeywordExtractor(directed=False, weighted=False, window_size=8)
            coreness = extractor_kw_cr.extract(' '.join(text))
            coreness_keyword = [x[0] for x in coreness]
            if len(coreness_keyword) > n:
                coreness_keyword = coreness_keyword[:n]


            '''centrality'''
            builder = GoWBuilder(directed=False, window_size=8)
            sentence_not_token = HGoW_generate.text_preprocessing_not_token(sentences)
            G1 = builder.compute_gow_from_document(sentence_not_token)
            g = G1.to_labeled_graph()

            '''keyBERT'''
            keybert = KeyBERT().extract_keywords(sentence_not_token)
            keybert_keyword = [x[0] for x in keybert]
            if len(keybert_keyword) > n:
                keybert_keyword = keybert_keyword[:n]

            # print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True))
            degree_centrality = nx.degree_centrality(g)
            degree_cen_keyword = [x[0] for x in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:n]]

            eigenvector_centrality = nx.eigenvector_centrality(g)
            eigenvector_cen_keyword = [x[0] for x in sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:n]]



            '''evaluation'''

            kgcore_eval, answer_eval = eval_set.evaluation_pre(kgcore_keyword, file_path+file_name.split('.')[0]+'.key')
            kgcore_f1 += f1_score(kgcore_eval, answer_eval)
            kgcore_precision += precision_score(kgcore_eval, answer_eval)
            kgcore_recall += recall_score(kgcore_eval, answer_eval)

            coreness_eval, answer_eval = eval_set.evaluation_pre(coreness_keyword, file_path + file_name.split('.')[0] + '.key')
            coreness_f1 += f1_score(coreness_eval, answer_eval)
            coreness_precision += precision_score(coreness_eval, answer_eval)
            coreness_recall += recall_score(coreness_eval, answer_eval)

            degree_cen_eval, answer_eval = eval_set.evaluation_pre(degree_cen_keyword, file_path + file_name.split('.')[0] + '.key')
            degree_cen_f1 += f1_score(degree_cen_eval, answer_eval)
            degree_cen_precision += precision_score(degree_cen_eval, answer_eval)
            degree_cen_recall += recall_score(degree_cen_eval, answer_eval)

            eigenvector_cen_eval, answer_eval = eval_set.evaluation_pre(eigenvector_cen_keyword, file_path + file_name.split('.')[0] + '.key')
            eigenvector_cen_f1 += f1_score(eigenvector_cen_eval, answer_eval)
            eigenvector_cen_precision += precision_score(eigenvector_cen_eval, answer_eval)
            eigenvector_cen_recall += recall_score(eigenvector_cen_eval, answer_eval)

            keybert_eval, answer_eval = eval_set.evaluation_pre(keybert_keyword, file_path + file_name.split('.')[0] + '.key')
            keybert_f1 += f1_score(keybert_eval, answer_eval)
            keybert_precision += precision_score(keybert_eval, answer_eval)
            keybert_recall += recall_score(keybert_eval, answer_eval)

    print('kgcore_f1 : ', kgcore_f1/size)
    print('coreness_f1 : ', coreness_f1/size)
    print('degree_cen_f1 : ', degree_cen_f1/size)
    print('eigenvector_cen_f1 : ', eigenvector_cen_f1/size)
    print('keybert_f1 : ', keybert_f1/size)

    print('kgcore_precision : ', kgcore_precision/size)
    print('coreness_precision : ', coreness_precision/size)
    print('degree_cen_precision : ', degree_cen_precision/size)
    print('eigenvector_cen_precision : ', eigenvector_cen_precision/size)
    print('keybert_precision : ', keybert_precision/size)

    print('kgcore_recall : ', kgcore_recall/size)
    print('coreness_recall : ', coreness_recall/size)
    print('degree_cen_recall : ', degree_cen_recall/size)
    print('eigenvector_cen_recall : ', eigenvector_cen_recall/size)
    print('keybert_recall : ', keybert_recall/size)

    print('size : ', size)
    write_data = [['kgcore_f1',kgcore_f1/size],[ 'coreness_f1',coreness_f1/size],['degree_cen_f1',degree_cen_f1/size],['eigenvector_cen_f1',eigenvector_cen_f1/size],['keybert_f1',keybert_f1/size],
                    ['kgcore_precision',kgcore_precision/size],['coreness_precision',coreness_precision/size],['degree_cen_precision',degree_cen_precision/size],['eigenvector_cen_precision',eigenvector_cen_precision/size],['keybert_precision',keybert_precision/size],
                    ['kgcore_recall',kgcore_recall/size],['coreness_recall',coreness_recall/size],['degree_cen_recall',degree_cen_recall/size],['eigenvector_cen_recall',eigenvector_cen_recall/size],['keybert_recall',keybert_recall/size]]

    df = pd.DataFrame(write_data)
    df.to_csv(f'{file_path}n=5_result.csv', index=False, header=False)