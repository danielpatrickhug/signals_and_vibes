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

rapidapi_host = config['free_news_rapid_api']['x-rapidapi-host']
rapidapi_key = config['free_news_rapid_api']['x-rapidapi-key']

# make request to news catcher api using rapid api
def get_free_news_data(topic):
    if type(topic) != str:
        topic = str(topic)
    url = "https://free-news.p.rapidapi.com/v1/search"
    querystring = {"q":f'{topic}' ,"lang":"en"}
    headers = {
        'x-rapidapi-host': rapidapi_host,
        'x-rapidapi-key': rapidapi_key
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    return response