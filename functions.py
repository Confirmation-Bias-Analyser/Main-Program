from anytree import Node, RenderTree, search
import re
import urllib
from urllib.parse import urlparse
from textblob import TextBlob
from sklearn.cluster import KMeans
import os
import networkx as nx
import psycopg2
from bs4 import BeautifulSoup
import pandas as pd
import requests
from pyvis.network import Network

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer()

from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
        
def setUpDB(command, url):
    """ create tables in the PostgreSQL database"""
    
    try:        
        # connect to the PostgreSQL server
        conn = psycopg2.connect(url, sslmode='require')
        cur = conn.cursor()
        
        # create table one by one
        cur.execute(command)
        
        # close communication with the PostgreSQL database server
        cur.close()
        
        # commit the changes
        conn.commit()
        conn.close()
        # print('done')
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        
def getData(command, url):    
    conn = psycopg2.connect(url, sslmode='require')
    
    try:
        df = pd.read_sql(command, conn)
        if conn is not None:
            conn.close()
        
        return df
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        
def create_Twitter_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers        
        
def getTweetsByUserID(user_id, header, max_results = 50):
    tweets_url = f'https://api.twitter.com/2/users/{user_id}/tweets?max_results={max_results}'
    return connect_to_endpoint(tweets_url, header)
    
def getSingleTweetInfo(tweetID, header):
    tweets_url = f'https://api.twitter.com/2/tweets?tweet.fields=created_at,conversation_id,in_reply_to_user_id,author_id,referenced_tweets&ids={tweetID}'
    return connect_to_endpoint(tweets_url, header)
    
def getTwitterUserInfo(username, header):
    user_result = f'https://api.twitter.com/2/users/by?usernames={username}&user.fields=created_at&expansions=pinned_tweet_id&tweet.fields=author_id,created_at'
    return connect_to_endpoint(user_result, header)
    
def getTweetsLikedByUser(userID, header, max_results = 30):
    user_result = f'https://api.twitter.com/2/users/{userID}/liked_tweets?max_results={max_results}'
    return connect_to_endpoint(user_result, header)
    
def getUsersLikesForTweet(tweetID, header):
    tweet_result = f'https://api.twitter.com/2/tweets/{tweetID}/liking_users'
    return connect_to_endpoint(tweet_result, header)

def connect_to_endpoint(url, headers, next_token = None):    
    response = requests.request("GET", url, headers = headers)
        
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
        
    return response.json()    

############# Data Collection

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

############# Confirmation Bias Model

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
        # sequence = re.sub(r'[0-9]+', '', sequence)
        sequence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sequence) # Remove the user name
        # sequence = sequence.replace(r'  ', '')
        # sequence = sequence.lower()
        sentences.append(sequence)

    return sentences

def calculateBias(dataset):
    count_positive_polarity_supportive = 0
    count_negative_polarity_supportive = 0
    count_positive_polarity_unsupportive = 0
    count_negative_polarity_unsupportive = 0

    for i in dataset.index.tolist():
        if dataset['vader_compound_score'].loc[i] > 0.35 and dataset['topic_cluster'].loc[i] == 1:
            count_positive_polarity_supportive += 1

        elif dataset['vader_compound_score'].loc[i] < -0.35 and dataset['topic_cluster'].loc[i] == 1:
            count_negative_polarity_supportive += 1

        elif dataset['vader_compound_score'].loc[i] > 0.35 and dataset['topic_cluster'].loc[i] == 0:
            count_positive_polarity_unsupportive += 1

        elif dataset['vader_compound_score'].loc[i] < -0.35 and dataset['topic_cluster'].loc[i] == 0:
            count_negative_polarity_unsupportive += 1
            
    total = count_positive_polarity_supportive + count_negative_polarity_supportive + count_positive_polarity_unsupportive + count_negative_polarity_unsupportive
    
    prob_D = (count_positive_polarity_supportive + count_negative_polarity_supportive)/total
    prob_D_prime = (count_positive_polarity_unsupportive + count_negative_polarity_unsupportive)/total
    result = {'P(D)': prob_D, 'P(D_p)': prob_D_prime}

    try:
        prob_D_H = count_positive_polarity_supportive / (count_positive_polarity_supportive + count_positive_polarity_unsupportive)
        prob_D_Hprime = count_negative_polarity_supportive / (count_negative_polarity_supportive + count_negative_polarity_unsupportive)
        
        if prob_D_H/prob_D_Hprime > 1:
            final_result = 1 / (prob_D_H/prob_D_Hprime)
            
        else:
            final_result = prob_D_H/prob_D_Hprime

        # return prob_D_H, prob_D_Hprime, final_result
        return final_result

    except:
        prob_D_H  = 0
        prob_D_Hprime = 0

        # return prob_D_H, prob_D_Hprime, 1
        return 1
        
def getClusters(allSentences, embedder, num_clusters = 2):
    corpus_embeddings = embedder.encode(allSentences)

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append([allSentences[sentence_id], sentence_id])

    return cluster_assignment        

def getSentimentalResults(sentence, vaderObject = sid_obj):
    textBlobResult = TextBlob(sentence)
    vaderResult = vaderObject.polarity_scores(sentence)
    compoundScore = vaderResult.pop('compound')
    
    overallResult = {'textblob_polarity': textBlobResult.sentiment.polarity, 
                     'textblob_subjectivity': textBlobResult.sentiment.subjectivity,
                     'vader_results': vaderResult,
                     'vader_compound_scores': compoundScore}

    return overallResult
    
def polarityDetermination(score, threshold = 0.35):
    if score > threshold:
        return 'POS'

    elif score < -1 * threshold:
        return 'NEG'

    else:
        return 'NEU'
    
def understandLinks(list_of_links):
    for i in list_of_links:
        if isinstance(i, str):
            print(i)
            
def definePolarity(score1, score2, threshold = 0.35):
    # Either both scores 1 & 2 are above threshold or when 1 of them is above threshold while the other is above 0
    if (score1 > threshold and score2 > threshold) or (score1 > 0 and score2 > threshold) or (score1 > threshold and score2 > 0):
        return 'POS'

    # Either both scores 1 & 2 are below threshold or when 1 of them is below threshold while the other is below 0
    elif (score1 < -1 * threshold and score2 < -1 * threshold) or (score1 < 0 and score2 < -1 * threshold) or (score1 < -1 * threshold and score2 < 0):
        return 'NEG'

    # Both scores are in the neutral range
    elif (score1 >= -1 * threshold and score1 <= threshold) and (score2 >= -1 * threshold and score2 <= threshold):
        return 'NEU'
        
    else:
        return 'UNKNOWN'
        
def defineSubjectivity(score1, score2, threshold = 0.5): # score 1 is model score while score 2 is textblob score
    if (score1 > threshold and score2 > threshold) or (score1 > threshold and score2 == threshold):
        return 'SUBJECTIVE'

    elif (score1 < threshold and score2 < threshold) or (score1 < threshold and score2 == threshold):
        return 'OBJECTIVE'
        
    else:
        return 'UNKNOWN'      

def getPolarityProportion(df):
    positivePolarity = len(df[df['overall_polarity'] == 'POS'])
    negativePolarity = len(df[df['overall_polarity'] == 'NEG'])
    
    return {'positive': positivePolarity/len(df), 'negative': negativePolarity/len(df)}
    
def getSubjectivityProportion(df):
    subjecitve = len(df[df['overall_subjectivity'] == 'SUBJECTIVE'])
    objecitve = len(df[df['overall_subjectivity'] == 'OBJECTIVE'])
    
    return {'subjecitve': subjecitve/len(df), 'objecitve': objecitve/len(df)}
    
def flagPotentialBias(df):
    result = []
    
    for index, row in df.iterrows():
        if row['overall_polarity'] == 'NEG' or row['overall_polarity'] == 'POS' or row['number_of_links'] > 0:
            result.append(1)
        
        else:
            result.append(0)
            
    return result

############# Data Visualisation

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