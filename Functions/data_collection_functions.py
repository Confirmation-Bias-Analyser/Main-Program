############# Data Collection #############

import psycopg2
import pandas as pd
import requests

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