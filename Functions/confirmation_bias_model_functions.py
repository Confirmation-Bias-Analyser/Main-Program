############# Confirmation Bias Model #############

from textblob import TextBlob
from sklearn.cluster import KMeans
import tensorflow as tf

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer()

from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-MiniLM-L6-v2')
	
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
    
def predictFromModel(model, tokeniser, data):
    tf_batch = tokeniser(data, max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

    return tf_predictions[:,1]    
    
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