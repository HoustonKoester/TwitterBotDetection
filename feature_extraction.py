import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

user_field_names = ['user_ID',      'createdAt',    'collectedAt',
                    'nFollowing',  'nFollowers',   'nTweets',
                    'nameLength',  'descriptionLength']

# returns a data frame with
#   user_ID, nFollowing, nFollowers, nTweets, nameLength, descriptionLength
def load_users(path_to_infile):
    return pandas.read_table(path_to_infile, names = user_field_names,
                             usecols = [0,3,4,5,6,7])

# returns a dict of the format
# { user_ID : [list of tweets] }
# TODO: If a user has no tweets or improperly formatted tweets,
#       they are useless, so we should remove them.
def load_tweets(path_to_infile, ignoreLines = []):
    tweet_dict = dict()

    ignored_lines = ignoreLines;

    lineNum = 0
    with open(path_to_infile) as infile:
        lines = infile.read().splitlines()
        for line in lines:
            if lineNum not in ignoreLines:
                try:
                    fields = line.split('\t') 
                    user_ID = fields[0]
                    tweet  = fields[2]

                    if user_ID in tweet_dict:
                        tweet_dict[user_ID].append(tweet)
                    else:
                        tweet_dict[user_ID] = [tweet]
                except:
                    ignored_lines.append(lineNum)
            lineNum += 1

    #return tweet_dict, ignored_lines
    return tweet_dict

# Some tweets may be highly similar and only differ in the content of a
# shortened URL, i.e., bit.ly/XXXXX compared to bit.ly/YYYYY. So, instead of
# comparing tweets directly we filter out stop words and compare word counts
# per tweet. Two tweets t1 and t2 where t1 != t2 are considered similar if
# <t1,t2> / ( |t1||t2| ) > threshold, i.e., if cos(t1,t2) > threshold.
# This function returns the percentage of a users tweets that are similar
# to another of their tweets.
def similar_tweets(user_tweets, threshold = 0.9):
    vector_cos = lambda a,b: (
                    np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))
                    if np.dot(a,a) != 0 and np.dot(b,b) != 0
                    else 0
   )

    similar_pairs = []

    counter = CountVectorizer()
    count = counter.fit_transform(user_tweets).toarray()
    for i in range(len(count)):
        for j in range(i+1,len(count)):
            t1 = count[i]
            t2 = count[j]
            if vector_cos(t1,t2) >= threshold:
                similar_pairs.append( (user_tweets[i], user_tweets[j]) )
                break

    n_similar = len(similar_pairs)
    percent_similar = n_similar/len(user_tweets)

    return percent_similar

def avg_number_of_mentions(user_tweets):
    num_mentions = sum([ t.count('@') for t in user_tweets ])
    return num_mentions/len(user_tweets)

# a tweet contains a link if 'http' or 'www' appears in the text of the tweet
def percent_tweets_with_links(user_tweets):
    num_tweets_with_links = len([t for t in user_tweets if 'http' in t or 'www' in t])
    return num_tweets_with_links/len(user_tweets)


def compute_features(tweet_dict, num_samples = 1000):
    USER_SAMPLE = np.random.choice(len(tweet_dict), num_samples, replace = False)

    USER_AVG_NUM_MENTIONS = []
    USER_PERCENT_SIMILAR_TWEETS = []
    USER_PERCENT_LINKS = []

    for n in USER_SAMPLE:
        user_ID = list(tweet_dict.keys())[n]

        avg_num_mentions = avg_number_of_mentions(tweet_dict[user_ID])
        percent_similar = similar_tweets(tweet_dict[user_ID])
        percent_links = percent_tweets_with_links(tweet_dict[user_ID])

        USER_AVG_NUM_MENTIONS.append(avg_num_mentions)
        USER_PERCENT_SIMILAR_TWEETS.append(percent_similar)
        USER_PERCENT_LINKS.append(percent_links)

#[user_avg_mentions, user_percent_links, user_percent_similar];

    return USER_SAMPLE, USER_AVG_NUM_MENTIONS, USER_PERCENT_SIMILAR_TWEETS, \
           USER_PERCENT_LINKS

# { user_ID: [number of followers over time] }
# not used
def load_following_series(path_to_infile):
    with open(path_to_infile, 'r') as infile:
        lines = infile.read().splitlines()

        return { l.split('\t')[0] : l.split('\t')[1].split(',') for l in lines }


