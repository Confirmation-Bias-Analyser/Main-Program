############# Data Visualisation #############

from anytree import Node, RenderTree, search
import networkx as nx
import requests
from pyvis.network import Network

def createNetworkGraph(conversation_tree, head_thread):
    G = nx.Graph()
    G.add_node(conversation_tree.name)

    for _, __, node in RenderTree(conversation_tree):
    
        try:
            G.add_edge(node.parent.name, node.name)

        except:
            if node.name == head_thread:
                continue

            G.add_edge(head_thread, node.name)

    return G
    
def createInterativeNetworkGraph(conversation_tree, head_thread, map_dict, scoreDict):
    G = Network("500px", "500px", notebook=True)
    
    if len(scoreDict) > 0:
        option = True
    else:
        option = False

    for _, __, node in RenderTree(conversation_tree):
        try:
            if option:
                G.add_node(node.name, label=node.name, color=map_dict[node.name], title='Score: ' + str(scoreDict[node.name]))
                
            else:
                G.add_node(node.name, label=node.name, color=map_dict[node.name])

        except:
            G.add_node(node.name, label=node.name)

    for _, __, node in RenderTree(conversation_tree):
    
        try:
            G.add_edge(node.parent.name, node.name)

        except:
            if node.name == head_thread:
                continue

            G.add_edge(head_thread, node.name)

    return G
    
def getColourNodes(conversationDF):
    polarity_map = {}
    subjectivity_map = {}
    potential_bias_map = {}

    for i in conversationDF.index.tolist():
        
        ########## Polarity detection results
        
        if conversationDF['overall_polarity'].loc[i] == 'POS':
            polarity_map[conversationDF['id'].loc[i]] = '#00FF80' # Green colour

        elif conversationDF['overall_polarity'].loc[i] == 'NEG':
            polarity_map[conversationDF['id'].loc[i]] = '#FF9999' # Light pink colour
        
        # Neutral class or Unknown
        else:
            polarity_map[conversationDF['id'].loc[i]] = '#FFFF00' # Yellow colour
   
        ########## Subjectivity detection results
        
        if conversationDF['overall_subjectivity'].loc[i] == 'SUBJECTIVE':
            subjectivity_map[conversationDF['id'].loc[i]] = '#A9A9A9' # Light grey colour

        elif conversationDF['overall_subjectivity'].loc[i] == 'OBJECTIVE':
            subjectivity_map[conversationDF['id'].loc[i]] = '#ADD8E6' # Light blue colour
           
        # Unknown class
        else:
            subjectivity_map[conversationDF['id'].loc[i]] = '#FF7F50' # Red orange colour
            
        ########## Potential Bias
        
        if conversationDF['potential_bias'].loc[i] == 1:
            potential_bias_map[conversationDF['id'].loc[i]] = 'red'

        elif conversationDF['potential_bias'].loc[i] == 0:
            potential_bias_map[conversationDF['id'].loc[i]] = 'black'
            
    return polarity_map, subjectivity_map, potential_bias_map