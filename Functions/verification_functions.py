############# Results Verification #############

from Functions.data_collection_functions import *
from Functions.confirmation_bias_model_functions import *
import re

def obtainTweetsAndLikes(userID, header):
    selectedUserTweets = getTweetsByUserID(userID, header, 100)
    userLikedTweets = getTweetsLikedByUser(userID, header, 100)
    allTweets = combineTweets([selectedUserTweets, userLikedTweets])

    return allTweets

def combineTweets(listOfTweets):
    allTweets_ = []

    for i in listOfTweets:
        for j in i['data']:
            allTweets_.append(j['text'])

    return allTweets_

def checkForRepliesToNews(textList):
    result = []
    for i in textList:
        reply = re.findall("(@[A-Za-z0-9]+)", i)

        if any(x in ['@MothershipSG', '@straits_times', '@ChannelNewsAsia', '@YahooSG'] for x in reply):
            result.append(i)

    return result

def calculateUserBias(tweetsData, embedder, defaultClusterSize = 3):
    user_tweets = []
    textblob_polarity_res = []
    textblob_subjectivity_res = [] 
    vader_compound_scores = []
    model_subjectivity_score = []
    clean_text = []

    for i in tweetsData:
        # repliedAccounts = re.findall("(@[A-Za-z0-9]+)", i)

        # if any(x in ['@MothershipSG', '@straits_times', '@ChannelNewsAsia'] for x in repliedAccounts):
        user_tweets.append(i)
        reply = cleanComments([i])
        clean_text.append(reply[0])
        sentimentalResults = getSentimentalResults(reply[0])

        textblob_polarity_res.append(sentimentalResults['textblob_polarity'])
        textblob_subjectivity_res.append(sentimentalResults['textblob_subjectivity'])
        vader_compound_scores.append(sentimentalResults['vader_compound_scores'])
        model_subjectivity_score.append(float(predictFromModel(model, tokenizer, reply)))

    overall_subjectivity = []
    for i in range(len(model_subjectivity_score)):
        overall_subjectivity.append(defineSubjectivity(model_subjectivity_score[i], textblob_subjectivity_res[i]))

    overall_polarity = []
    for i in range(len(textblob_polarity_res)):
        overall_polarity.append(definePolarity(textblob_polarity_res[i], vader_compound_scores[i]))

    cluster = getClusters(clean_text, embedder, defaultClusterSize)

    df = pd.DataFrame(list(zip(user_tweets, textblob_polarity_res, textblob_subjectivity_res, vader_compound_scores, model_subjectivity_score, overall_subjectivity, overall_polarity, cluster)),
                      columns =['tweet', 'textblob_polarity', 'textblob_subjectivity', 'vader_compound_score', 'model_subjectivity', 'overall_subjectivity', 'overall_polarity', 'topic_cluster'])
    
    return df