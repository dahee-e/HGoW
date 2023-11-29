import networkx as nx
import time
import os
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import re, string
import spacy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_set_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content



def text_preprocessing(sentences):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    STOPWORDS = set(stopwords.words('english'))
    result = []
    sentences = sent_tokenize(sentences)
    for sentence in sentences:
        text = re.sub(r'\[[0-9]*\]', ' ', sentence)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\t', ' ', text)
        text = re.sub(r"\s+$", "", text)

        data = []
        doc = nlp(text)
        for token in doc:
            if token.text not in STOPWORDS and token.text not in string.punctuation:
                if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON') and (token.pos_ != 'SPACE'):
                    data.append(token.lemma_)
        result.append(data)
    return result

def text_preprocessing_not_token(sentences):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    STOPWORDS = set(stopwords.words('english'))
    result = ""
    sentences = sent_tokenize(sentences)
    for sentence in sentences:
        text = re.sub(r'\[[0-9]*\]', ' ', sentence)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\t', ' ', text)
        text = re.sub(r"\s+$", "", text)

        data = []
        doc = nlp(text)
        for token in doc:
            if token.text not in STOPWORDS and token.text not in string.punctuation:
                if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON') and (token.pos_ != 'SPACE'):
                    data.append(token.lemma_)
        result+= " ".join(data)+" "

    return result

def load_hypergraph(text):
    hypergraph = nx.Graph()  # Create an empty hypergraph
    E = list()
    for line_number, line in enumerate(text, start=1):
        nodes = {node for node in line}
        nodes = {x for x in nodes}
        hyperedge = set(nodes)  # Use frozenset to represent the hyperedge
        E.append(hyperedge)
        for node in nodes:
            if node not in hypergraph.nodes():
                hypergraph.add_node(node, hyperedges=list())  # Add a node for each node
            hypergraph.nodes[node]['hyperedges'].append(hyperedge)  # Add the hyperedge to the node's hyperedge set

    return hypergraph, E

