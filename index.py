import networkx as nx
import time
import os
import argparse
from collections import Counter

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.value = {}

    def postorder(self):
        for child in self.children:
            child.postorder()
        if not self.children:
            print(self.name, 'value:', self.value)

    def leaf(self):
        for i in range(len(self.children)):
            leaf = self.children[i].children[-1]
            print(leaf.name, 'value:', leaf.value)
    def leaf_count(self):
        set_of_words = []
        for i in range(len(self.children)):
            leaf = self.children[i].children[-1]
            set_of_words.append(list(leaf.value))
        return set_of_words


def load_hypergraph(file_path):
    hypergraph = nx.Graph()  # Create an empty hypergraph
    E = list()

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # Use set to ignore duplicate values in each line and strip whitespace from node names
            nodes = {node.strip() for node in line.strip().split(',')}
            nodes = {int(x) for x in  nodes}
            hyperedge = set(nodes)  # Use frozenset to represent the hyperedge
            E.append(hyperedge)
            for node in nodes:
                if node not in hypergraph.nodes():
                    hypergraph.add_node(node, hyperedges=list())  # Add a node for each node
                hypergraph.nodes[node]['hyperedges'].append(hyperedge)  # Add the hyperedge to the node's hyperedge set

    return hypergraph, E



"""hypergraph에 속한 각각 노드에 대해 이웃 노드들과, 이웃 노드들과의 co-occurence를 반환하는 함수"""
def neighbour_count_map(hypergraph, v,g):
    neighbor_counts = {}
    for hyperedge in hypergraph.nodes[v]['hyperedges']:
        # Increment the count for each neighbor in the hyperedge
        for neighbor in hyperedge:
            if neighbor != v:
                neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    filtered_neighbors = {neighbor: count for neighbor, count in neighbor_counts.items() if count >= g}
    return filtered_neighbors


""" 고정된 g 값에 대해 k를 증가시키면서 모든 (k,g)-core를 찾는 함수"""
def enumerate_kg_core_fixing_g(hypergraph, g):
    H = set(hypergraph.nodes)
    S = []
    for k in range(1,len(hypergraph.nodes)):
        if len(H) <= k:
            break
        while True:
            if len(H) <= k:
                break
            changed = False
            nodes = H.copy()
            for v in nodes:
                map = neighbour_count_map(hypergraph,v,g)
                map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
                if len(map) < k:
                    H -= {v}
                    changed = True
            if not changed:
                S.append(H.copy())
                break

    return S

# def enumerate_kg_core_fixing_k(hypergraph, E, k):
#     H = set(hypergraph.nodes)
#     S = []
#     for g in range(1,len(E)):
#         if len(H) <= k:
#             break
#         while True:
#             if len(H) <= k:
#                 break
#             changed = False
#             nodes = H.copy()
#             for v in nodes:
#                 map = neighbour_count_map(hypergraph,v,g)
#                 map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
#                 if len(map) < k:
#                     H -= {v}
#                     changed = True
#             if not changed and len(H) != 0:
#                 S.append(H.copy())
#                 break
#     return S

""" naive index construction"""
def naive_index_construction(hypergraph,E):
    T = TreeNode("root")
    for g in range(0,len(E)):
        S = enumerate_kg_core_fixing_g(hypergraph,(g+1))
        if len(S) == 0:
            break
        T.children.append(TreeNode(g+1))
        for s in range(len(S)):
            T.children[g].children.append(TreeNode((s+1,g+1)))
            T.children[g].children[s].value = S[s]
    return T


# def naive_index_construction(hypergraph,E):
#     T = TreeNode("root")
#     for k in range(1, len(hypergraph.nodes)):
#         S = enumerate_kg_core_fixing_k(hypergraph, E, k)
#         if len(S) == 0:
#             break
#         T.children.append(TreeNode(k))
#         for s in range(len(S)):
#             T.children[k-1].children.append(TreeNode((s+1,k)))
#             T.children[k - 1].children[s].value = S[s]
#     return T














