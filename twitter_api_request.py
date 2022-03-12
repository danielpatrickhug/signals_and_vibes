import requests
import json
import os
import yaml
import base64
import tweepy

def get_config():
    """
    Get the configuration.
    :return:
    """
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = get_config()

API_KEY = config['twitter_api']['TWITTER_API_KEY']
API_SECRET = config['twitter_api']['TWITTER_API_SECRET']
BEARER_TOKEN = config['twitter_api']['TWITTER_BEARER_TOKEN']


auth = tweepy.OAuth2BearerHandler(str(BEARER_TOKEN))
client = tweepy.Client(auth)

#make curl request twiter api version 2
def get_tweet_by_id(tweet_id):
    """
    Get the user timeline of a user_id.
    :param user_id:
    :param count:
    :return:
    """
    url = 'https://api.twitter.com/2/tweets/{}'.format(tweet_id)
    #pass bearer token
    headers = { 'Authorization': 'Bearer {}'.format(BEARER_TOKEN) }
    parameters = {'expansions': 'author_id'}
    response = requests.get(url, headers=headers, params=parameters)
    return response.json()

def get_users_followers_id(user_id):
    """
    Get the user timeline of a user_id.
    :param user_id:
    :param count:
    :return:
    """
    url = "https://api.twitter.com/2/users/{}/followers".format(user_id)
    #pass bearer token
    headers = { 'Authorization': 'Bearer {}'.format(BEARER_TOKEN) }
    parameters = {"user.fields": "created_at"}
    response = requests.get(url, headers=headers, params=parameters)
    return response.json()['data']

def get_user_id_by_name(account_name):
    usernames = f"usernames={account_name}"
    user_fields = "user.fields=description,created_at"
    url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
    headers = { 'Authorization': 'Bearer {}'.format(BEARER_TOKEN) }
    response = requests.get(url, headers=headers)
    return response.json()['data'][0]['id']

def get_timeline_by_user_id(user_id, n=1):
    url = "https://api.twitter.com/2/users/{}/tweets".format(user_id)
    headers = { 'Authorization': 'Bearer {}'.format(BEARER_TOKEN) }
    params = {"tweet.fields": "created_at"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()['data'][:n]


