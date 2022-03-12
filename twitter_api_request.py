import requests
import json
import os
import yaml

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

def configure_twitter_api():
    """
    Configure the Twitter API.
    :return:
    """
    url = 'https://api.twitter.com/oauth2/token'
    params = {'grant_type': 'client_credentials'}
    headers = {
        'Authorization': f'Basic {API_KEY}:{API_SECRET}',
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }
    response = requests.post(url, params=params, headers=headers)
    return response.json()['access_token']


def get_user_timeline(user_id, count=200):
    """
    Get the user timeline of a user_id.
    :param user_id:
    :param count:
    :return:
    """
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
    params = {'user_id': user_id, 'count': count}
    response = requests.get(url, params=params)
    return response.json()

def get_user_id_by_account_name(account_name):
    """
    Get the user_id of a user by their account name.
    :param account_name:
    :return:
    """
    url = 'https://api.twitter.com/1.1/users/show.json'
    params = {'screen_name': account_name}
    response = requests.get(url, params=params)
    return response.json()['id']

def get_user_followers(user_id):
    url = 'https://api.twitter.com/1.1/followers/ids.json'
    params = {'user_id': user_id}
    response = requests.get(url, params=params)
    return response.json()['ids']

def get_top_n_hashtag_tweets(hashtag, n=20):
    url = 'https://api.twitter.com/1.1/search/tweets.json'
    params = {'q': hashtag, 'count': n}
    response = requests.get(url, params=params)
    return response.json()

print(API_KEY)
