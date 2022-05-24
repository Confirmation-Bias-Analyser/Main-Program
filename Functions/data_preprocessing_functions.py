############# Data Processing #############

from anytree import Node, RenderTree, search
import re
import urllib
from urllib.parse import urlparse
import os

def getTweetComments(conversation_data):
    conversation_dict = {'id':[], 'timestamp':[], 'reply_to':[], 'comment':[]}

    for i in conversation_data:
        print('User ID:', i['id'], 
              'Time:', i['user']['created_at'])
        print('In reply to:', i['in_reply_to_status_id'])
        print(i['text'], '\n')

        conversation_dict['id'].append(i['id'])
        conversation_dict['timestamp'].append(i['user']['created_at'])
        conversation_dict['reply_to'].append(i['in_reply_to_status_id'])
        conversation_dict['comment'].append(i['text'])

    return conversation_dict

def getLinks(string):
    urls = re.findall("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", string)
    links = ''

    for url in urls:
        try:
            opener = urllib.request.build_opener()
            request = urllib.request.Request(url)
            response = opener.open(request)
            actual_url = response.geturl()
                      
            if '](' in actual_url:
                actual_url = actual_url.split('](')[0]
          
            links += actual_url + ';'
            
            
        except:
            if '](' in url:
                url = url.split('](')[0]
          
            links += url + ';'

    return links

def getURLfromList(url):
    if ';' in url:
        url = url.split(';')[:-1]
        result = []
    
        for i in url:
            result.append(urlparse(i).hostname)
        
        return result

    else:
        return ''

def printDetailsPHEME(threads, data):
    rumours = 0
    non_rumours = 0

    for i in threads:
        path = '/content/all-rnr-annotated-threads/' + i
        print(i)

        for j in os.listdir(path):
          
            for k in data:

                for l in os.listdir(path + '/' + k):
                    if k == data[0] and l[0] != '.':
                        non_rumours += 1

                    elif k == data[1] and l[0] != '.':
                        rumours += 1

    print('Rumours:', rumours)
    print('Non-rumours:', non_rumours)
    print()
	
def traceConversation(dataframe, tree, node, printGraphOption = True):
    children_nodes_list = getAllChildNodes(tree, node, [])

    print('\n\n')
    new = search.find_by_attr(tree, node)
    
    if printGraphOption:
        printGraph(new)

    return dataframe[(dataframe['reply_to'].isin(children_nodes_list)) | (dataframe['id'].isin(children_nodes_list + [node]))], new

def getAllChildNodes(tree, node, children_nodes_list):
    children_nodes = search.find_by_attr(tree, node).children
    
    for i in children_nodes:
        children_nodes_list.append(i.name)
        
        if i.children != None:
            getAllChildNodes(tree, i.name, children_nodes_list)
            
        else:
            return
            
    return children_nodes_list

def cleanComments(comments_array):
    sentences = []

    for i in comments_array:
        sequence = i.replace('\n', ' ') # Remove new line characters
        sequence = sequence.replace('\.', '')
        sequence = sequence.replace('.', '')
        sequence = sequence.replace(",", " ")
        sequence = sequence.replace("'", " ")
        sequence = sequence.replace('\\', '')
        sequence = sequence.replace('\'s', '')
        sequence = sequence.replace('&gt;', '') # Remove ampersand
        sequence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sequence) # Remove the user name
        sentences.append(sequence)

    return sentences
    
def createTweetsTree(dictionary, tree_root):
    for key, item in dictionary.items():
        child = Node(key, parent=tree_root)

        if len(dictionary[key]) != 0:      
            createTweetsTree(dictionary[key], child)

        else:
            continue

def make_map(list_child_parent):
    has_parent = set()
    all_items = {}
    
    for child, parent in list_child_parent:
        if parent not in all_items:
            all_items[parent] = {}
            
        if child not in all_items:
            all_items[child] = {}
        
        all_items[parent][child] = all_items[child]
        has_parent.add(child)

    result = {}
    
    for key, value in all_items.items():
        if key not in has_parent:
            result[key] = value
    
    return result

def printGraph(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))    