from transformers import BertTokenizer, TFBertForSequenceClassification, InputExample, InputFeatures
import pandas as pd
import numpy as np
import random
import re
from anytree.exporter import DotExporter
from graphviz import Source, render
import warnings
warnings.filterwarnings("ignore")

# The shutil module offers a number of high-level operations on files and collections of files.
import os
import shutil

import sys
sys.path.append('/Functions/')
from Functions.confirmation_bias_model_functions import *
from Functions.data_collection_functions import *
from Functions.data_preprocessing_functions import *
from Functions.visualisation_functions import *
from Functions.verification_functions import *

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS

from dash import Dash, html, Input, Output, dash_table, dcc
from dash.dependencies import Input, Output
from IPython.core.display import display, HTML

with open('Authentication/twitter_bearer_token.txt', 'r', encoding="utf8") as f:
    token = f.read()

header = create_Twitter_headers(token)

with open('Authentication/database_uri.txt', 'r', encoding="utf8") as f:
    uri = f.read()


print("\n************ Welcome to Confirmation Bias Analyser ************\n")

# # Data Collection

# ## Read the data and present as a dataframe

overall_results_str = ""

tweets_for_analysis = ['1522931750451617793', '1507922082683793408', '553553331671408641']

for i in range(len(tweets_for_analysis)):
    print(i+1, tweets_for_analysis[i])

tweetOption = int(input("Please indicate the tweet to analyse. ")) - 1

print(f'\nChosen {tweets_for_analysis[tweetOption]}\n')

if tweetOption == 0 or tweetOption == 1:
    conversationID = tweets_for_analysis[tweetOption]

    query = '''
        select * from comments_for_analysis where conversation_id = '%s'
    '''% conversationID
    
    df = getData(query, uri)

    parent = df['head_id'][0]
    
    overall_results_str += 'About the tweet:' + '\n' + getSingleTweetInfo(conversationID, header)['data'][0]['text'] + '\n' + 'Number of comments: ' + str(len(df)) + '\n'
    print(overall_results_str)

elif tweetOption == 2:
    parent = tweets_for_analysis[tweetOption]

    query = '''
        select * from pheme_dataset_for_analysis where head_id = '%s'
    '''% parent

    df = getData(query, uri)

    overall_results_str += 'Number of comments:' + str(len(df)) + '\n'
    print(overall_results_str)
    
    
# Create directory to save all results
newDirectory = f"Results/{tweets_for_analysis[tweetOption]}/"

if tweets_for_analysis[tweetOption] not in os.listdir("Results/"):
    os.mkdir(newDirectory)
    
df['url'] = df['comment'].apply(lambda x: getLinks(x))
df['link_title'] = df['url'].apply(lambda x: getURLfromList(x))
df.head()    


# In[5]:


# with open(mainDirectory + 'database_uri.txt', 'r', encoding="utf8") as f:
#     uri = f.read()

# socialMedia = ['Reddit', 'Twitter', 'PHEME Dataset']

# for i in range(len(socialMedia)):
#     print(i, socialMedia[i])

# socialMediaOption = 1

# print(f'\nChosen {socialMedia[socialMediaOption]}\n')

# if socialMedia[socialMediaOption] == 'Reddit':
#     parent = 'rmqevj'
#     query = '''
#         select * from reddit_posts_for_analysis where head_id = '%s'
#     '''% parent
#     df = getData(query, uri)
    
# elif socialMedia[socialMediaOption] == 'Twitter':
#     # conversationID = '1522931750451617793'
#     conversationID = '1507922082683793408'

#     query = '''
#         select * from comments_for_analysis where conversation_id = '%s'
#     '''% conversationID
    
#     df = getData(query, uri)

#     parent = df['head_id'][0]

#     print('About the tweet:')
#     print(getSingleTweetInfo(conversationID, header)['data'][0]['text'])

#     print('Number of comments:', len(df))

# elif socialMedia[socialMediaOption] == 'PHEME Dataset':
#     parent = '553553331671408641'
#     query = '''
#         select * from pheme_dataset_for_analysis where head_id = '%s'
#     '''% parent
#     df = getData(query, uri)

# df['url'] = df['comment'].apply(lambda x: getLinks(x))
# df['link_title'] = df['url'].apply(lambda x: getURLfromList(x))
# df.head()


# ## Construction of tree of comments

root = Node(parent)

input_list = [] 

item_count = df['reply_to'].value_counts().to_dict()

for i in range(len(df['id'].tolist())):
    try:
        if df['id'].loc[i] != df['reply_to'].loc[i]:
            input_list.append((df['id'].loc[i], df['reply_to'].loc[i]))

    except:
        continue

output_dict = make_map(input_list)
createTweetsTree(output_dict[parent], root)


# # Sentiment Analysis

pred_sentences = cleanComments(df['comment'])

tokenizer = BertTokenizer.from_pretrained("subjectivity_tokenizer/")
model = TFBertForSequenceClassification.from_pretrained("saved_subjectivity_model/")

textblob_polarity = []
textblob_subjectivity = []
vader_results = []
vaderCompoundScores = []
model_subjectivity_result = predictFromModel(model, tokenizer, pred_sentences)

for i in pred_sentences:
    result = getSentimentalResults(i)
    textblob_polarity.append(result['textblob_polarity'])
    textblob_subjectivity.append(result['textblob_subjectivity'])
    vader_results.append(result['vader_results'])
    vaderCompoundScores.append(result['vader_compound_scores'])


# ## Saving of results from sentiment analysis into the same dataframe

df['number_of_links'] = df['link_title'].apply(lambda x: len(x))

# Polarity
df['textblob_polarity'] = textblob_polarity
df['vader_compound_score'] = vaderCompoundScores
df['vader_polarity'] = df['vader_compound_score'].apply(lambda x: polarityDetermination(x))

overall_polarity = []
overall_polarity_scores = {}

for i in range(len(textblob_polarity)):
    overall_polarity.append(definePolarity(textblob_polarity[i], vaderCompoundScores[i]))
    overall_polarity_scores[df['id'][i]] = [textblob_polarity[i], vaderCompoundScores[i]]

df['overall_polarity'] = overall_polarity

# Subjectivity
df['model_subjectivity'] = model_subjectivity_result
df['textblob_subjectivity'] = textblob_subjectivity

overall_subjectivity = []
overall_subjectivity_scores = {}

for i in range(len(model_subjectivity_result)):
    overall_subjectivity.append(defineSubjectivity(model_subjectivity_result[i], textblob_subjectivity[i]))
    overall_subjectivity_scores[df['id'][i]] = [float(model_subjectivity_result[i]), textblob_subjectivity[i]]

df['overall_subjectivity'] = overall_subjectivity

df['vader_sentiment'] = vader_results
df['topic_cluster'] = getClusters(pred_sentences, embedder)

df['potential_bias'] = flagPotentialBias(df)

df.to_csv(f'{newDirectory}/sentiment_result.csv', index=False)

# # Confirmation Bias Analysis

head_thread = parent #input('Enter a comment to look at the replies. ')
conversationDF, conversationTree = traceConversation(df, root, head_thread, False)

about_str = "\nDetails of tweet\n" + "Existence of links:\n"
checkLink = False

for i in df['link_title']:
    if isinstance(i, list):
        about_str += str(i) + '\n'
        checkLink = False

if checkLink:
    about_str += "No links" + '\n'

else:
    print("\nThe exact link in the comments:")
    about_str += "\nThe exact link in the comments:" + '\n'
    
    for i in df['url']:
        if len(i) > 0:
            about_str += str(i) + '\n'

print(about_str)
overall_results_str += about_str

print("Conversation Tree")
printGraph(root)
DotExporter(root).to_dotfile(f'{newDirectory}/tree_of_comments.dot')
render('dot', 'jpg', f'{newDirectory}/tree_of_comments.dot')

# ## Results of confirmation bias analysis

overall_results_str += "\nConfirmation Bias Analysis\n"

cb_result_str = "\nConfirmation Bias Score for Entire Conversation:" + str(calculateBias(conversationDF)) + '\n' + "Number of potentially bias comments:" + str(len(conversationDF[conversationDF['potential_bias'] == 1]))
print(cb_result_str)
overall_results_str += cb_result_str

# ## List of potentially bias comments

overall_results_str += "\n\nPotentially biased comments (user id and comment):\n"
for index, row in conversationDF[conversationDF['potential_bias'] == 1].iterrows():
    overall_results_str += str(row['id']) + str(row['comment']) + '\n'
#     print(row['id'], row['comment'])

# # Results Verification

# ## Get the Replies and Retweets of Potentially Bias Users

replies_retweets_result_str = ""

if tweetOption == 0:
    print("Number of potentially bias users:", len(conversationDF[conversationDF['potential_bias'] == 1]['user_id'].unique()), '\n')

    for i in conversationDF[conversationDF['potential_bias'] == 1]['user_id']:
        allUserTweets = combineTweets([getTweetsLikedByUser(i, header, 100)])

        sub_heading = '*********** ' + i + ' ***********'
        replies_retweets_result_str += sub_heading + '\n'
        print(sub_heading)

        for j in checkForRepliesToNews(allUserTweets):
            print(j)
            replies_retweets_result_str += j + '\n'

        replies_retweets_result_str += '\n'
        print()

elif tweetOption == 1:
    print("Number of potentially bias users:", len(conversationDF[conversationDF['potential_bias'] == 1]['id'].unique()), '\n')

    suspectedBiasUsers = ['CardanoAdonis', 'PLafala', 'Erwin_Dawson', 'Coldcappuccino9', '_5andman_', 'Actarus_dEuphor', 'ResenT___', 'mkggoh', 'slowpokemax']

    for i in suspectedBiasUsers:
        userID = getTwitterUserInfo(i, header)['data'][0]['id']
        allUserTweets = combineTweets([getTweetsByUserID(userID, header, 100)])

        sub_heading = '*********** ' + i + ' ***********'
        replies_retweets_result_str += sub_heading + '\n'
        print(sub_heading)

        for j in checkForRepliesToNews(allUserTweets):            
            print(j)
            replies_retweets_result_str += j + '\n'

        replies_retweets_result_str += '\n'
        print()

if len(replies_retweets_result_str) > 0:
    with open(f'{newDirectory}/replies_and_retweets.txt', 'w', encoding="utf-8") as f:
        f.write(replies_retweets_result_str)


# ### Get the Liked Tweets by Potentially Bias Users

liked_tweets_result_str = ""

if tweetOption == 0:
    for i in conversationDF[conversationDF['potential_bias'] == 1]['user_id']:
        allUserTweets = combineTweets([getTweetsLikedByUser(i, header, 100)])

        sub_heading = '*********** ' + i + ' ***********'
        liked_tweets_result_str += sub_heading + '\n'
        print(sub_heading)

        for j in checkForRepliesToNews(allUserTweets):
            print(j)
            liked_tweets_result_str += j + '\n'

        liked_tweets_result_str += '\n'
        print()

elif tweetOption == 1:
    for i in suspectedBiasUsers:
        userID = getTwitterUserInfo(i, header)['data'][0]['id']
        allUserTweets = combineTweets([getTweetsLikedByUser(userID, header, 100)])

        sub_heading = '*********** ' + i + ' ***********'
        liked_tweets_result_str += sub_heading + '\n'
        print(sub_heading)

        for j in checkForRepliesToNews(allUserTweets):
            print(j)
            liked_tweets_result_str += j + '\n'

        liked_tweets_result_str += '\n'
        print()

if len(liked_tweets_result_str) > 0:
    with open(f'{newDirectory}/liked_tweets.txt', 'w', encoding="utf-8") as f:
        f.write(liked_tweets_result_str)

# # Results Visualisation

# ## Visualise through graph

polarity_map, subjectivity_map, pb_map = getColourNodes(conversationDF)

net = createInterativeNetworkGraph(conversationTree, head_thread, polarity_map, overall_polarity_scores)
net.show(f'{newDirectory}/polarity_scores_graph.html')
print('Polarity Scores')
print('Legend: Green - Positive polarity,', 'Pink - Negative polarity,', 'Yellow - Neutral or Unknown')


net = createInterativeNetworkGraph(conversationTree, head_thread, subjectivity_map, overall_subjectivity_scores)
net.show(f'{newDirectory}/subjectivity_scores_graph.html')
print('Subjectivity Scores')
print('Legend: Light Grey - Subjective comment,', 'Light Blue - Objective comment,', 'Orange - Unknown')


net = createInterativeNetworkGraph(conversationTree, head_thread, pb_map, [])
net.show(f'{newDirectory}/potential_bias_users_graph.html')
print('Legend: Red - Potential Bias,', 'Black - Potential Unbias')


# ## Other additional visualisation

# Let's count the number of tweets by sentiments
sentiment_counts = conversationDF.groupby(['overall_polarity']).size()
print(sentiment_counts)

topic_cluster_counts = conversationDF.groupby(['topic_cluster']).size()
print(topic_cluster_counts)

fig = plt.figure(figsize=(12,12), dpi=100)

ax1 = fig.add_subplot(211)
ax1.title.set_text("Results of polarity detection")
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=270)

ax2 = fig.add_subplot(212)
ax2.title.set_text("Results of text clustering")
ax2.pie(topic_cluster_counts.values, labels=topic_cluster_counts.index, autopct='%1.1f%%', startangle=270)

fig.savefig(f'{newDirectory}/pie_charts_polarity_and_text_clustering.png')

# Wordcloud with positive tweets
positive_tweets = []
for i in conversationDF[conversationDF["overall_polarity"] == 'POS']['comment'].tolist():
    positive_tweets.append(re.sub("(@[A-Za-z0-9]+)", "", i))

if len(positive_tweets) != 0:
    stop_words = ["https", "co", "RT"] + list(STOPWORDS)
    positive_wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white", stopwords = stop_words).generate(str(positive_tweets))

# Wordcloud with negative tweets
negative_tweets = []
for i in conversationDF[conversationDF["overall_polarity"] == 'NEG']['comment'].tolist():
    negative_tweets.append(re.sub("(@[A-Za-z0-9]+)", "", i))

if len(negative_tweets) != 0:
    stop_words = ["https", "co", "RT"] + list(STOPWORDS)
    negative_wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white", stopwords = stop_words).generate(str(negative_tweets))

    
fig = plt.figure(figsize=(6,6), dpi=100)

ax1 = fig.add_subplot(211)
ax1.title.set_text("Positive comments - Wordcloud")
ax1.axis("off")
ax1.imshow(positive_wordcloud, interpolation="bilinear")

ax2 = fig.add_subplot(212)
ax2.title.set_text("Negative comments - Wordcloud")
ax2.axis("off")
ax2.imshow(negative_wordcloud, interpolation="bilinear")

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

fig.savefig(f'{newDirectory}/wordcloud_for_positive_and_negative_sentiments.png')


with open(f'{newDirectory}/overall_results.txt', 'w', encoding="utf-8") as f:
    f.write(overall_results_str)


# scoreLists = ['number_of_links', 'vader_compound_score', 'textblob_polarity', 'textblob_subjectivity', 'model_subjectivity', 'overall_subjectivity', 'overall_polarity', 'topic_cluster', 'potential_bias']
# df_for_visualisation = conversationDF[['id','timestamp','reply_to','comment'] + scoreLists]

# app = Dash(__name__)

# app.layout = html.Div([
#     html.H4('Conversation'),
#     html.P(id='table_out'),
#     dash_table.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} 
#                  for i in ['id'] + scoreLists],
#         data=df_for_visualisation.to_dict('records'),
#         style_cell={
#             'textAlign': 'center',
#         },
#         style_as_list_view=True,
#         style_header=dict(backgroundColor="paleturquoise"),
#         style_data=dict(backgroundColor="lavender"),
#     ), 
# ])

# @app.callback(
#     Output('table_out', 'children'), 
#     Input('table', 'active_cell'))

# def update_graphs(active_cell):
#     if active_cell:
#         identifier = df_for_visualisation.iloc[active_cell['row']]['id']
#         comment = df_for_visualisation.iloc[active_cell['row']]['comment']

#         return f"Selected ID: {identifier}, Comment by user: \"{comment}\"" #, {score}"
#     return "Click the table"

# app.run_server(debug=True)#mode='inline'