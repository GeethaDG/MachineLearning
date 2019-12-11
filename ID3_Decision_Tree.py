"""
Parameters in command line
1 - Input/Data File Name
2 - Output File Name
"""
import pandas as pd
import numpy as np
import sys as s
import math as m
import xml.etree.cElementTree as ET

cmd_line = s.argv[1:]
for each in range(0, len(cmd_line)):
    if cmd_line[each] == '--data':
        in_data_file = cmd_line[each + 1]
    elif cmd_line[each] == '--output':
        out_file = cmd_line[each + 1]

# Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv(in_data_file, header=None)
for i in range(0, len(dataset.columns) - 1):
    dataset.rename(columns={dataset.columns[i]: 'att' + str(i)}, inplace=True)
dataset.rename(columns={dataset.columns[i + 1]: 'target'}, inplace=True)
space_size='global'
space_size=len(dataset.target.unique())
count='global'
count=0
tree = 'global'

def entropy(target):
    elements, counts = np.unique(target, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * m.log(counts[i] / np.sum(counts), space_size) for i in range(len(elements))])
    return entropy

def information_gain(feature,curr_feature_dataset,target_name):
    node_entropy=entropy(curr_feature_dataset[target_name])
    vals, counts = np.unique(curr_feature_dataset[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(curr_feature_dataset.where(curr_feature_dataset[feature] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    info_gain=node_entropy-weighted_entropy
    return info_gain,node_entropy


def ID3_decision_tree(original_dataset,feature_dataset,curr_dataset,target_name,features,recur_count,best_feature_value,best_feature_in, xml_doc):
    recur_count=recur_count+1
    info_gain_node_entropy_tuple=[information_gain(feature,curr_dataset,target_name) for feature in features]
    infor_gain=[0]*len(info_gain_node_entropy_tuple)
    node_entropy=[0]*len(info_gain_node_entropy_tuple)
    for i in range(len(info_gain_node_entropy_tuple)): infor_gain[i],node_entropy[i]=info_gain_node_entropy_tuple[i]
    best_feature_index = np.argmax(infor_gain)
    best_feature=features[best_feature_index]
    best_feature_values=np.unique(curr_dataset[best_feature])
    if xml_doc == None:
        xml_doc=ET.Element('tree', entropy=str(entropy_value))
    elif(node_entropy[0]==0):
        str1=''.join(str(e) for e in (np.unique(curr_dataset[target_name])))
        ET.SubElement(xml_doc, 'node', entropy=str(node_entropy[0]), value=best_feature_value,feature=best_feature_in).text=str1
        return xml_doc
    else:
        xml_doc=ET.SubElement(xml_doc,'node',entropy=str(node_entropy[0]),value=best_feature_value,feature=best_feature_in)
    for i in range (len(np.unique(curr_dataset[best_feature]))):
        feature_dataset=curr_dataset.where(curr_dataset[best_feature] == best_feature_values[i]).dropna()
        ID3_decision_tree(original_dataset,curr_dataset,feature_dataset,target_name,feature_dataset.columns[:-1],recur_count,best_feature_values[i],best_feature,xml_doc)
    return xml_doc

entropy_value = entropy(dataset.target)
xml_doc = None
xml_doc = ID3_decision_tree(dataset,dataset,dataset,'target',dataset.columns[:-1],count,'None','None', xml_doc)
tree = ET.ElementTree(xml_doc)
tree.write(out_file)

