import requests
import csv
import pandas as pd
import praw
import re
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from dotenv import load_dotenv
import os
import firebase_admin
from firebase_admin import credentials, storage
from tqdm import tqdm
import certifi
import prawcore

os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

# Download necessary NLTK data
nltk.download('vader_lexicon')


df = pd.read_csv('Stocks.csv')
stock_data = df.head(1000)['Symbol'].tolist()

# initialize Firebase

cred = credentials.Certificate("firebaseCredentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ryansdailynews-7df9d.appspot.com'
})

def data_procurement():
    global stock_data
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0",
    }

    # URLs for NYSE and NASDAQ
    urls = {
        'nasdaq': "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq"
    }

    combined_data = []

    for exchange, url in urls.items():
        r = requests.get(url, headers=headers)
        j = r.json()

        table = j['data']['table']
        table_headers = table['headers']
        rows = table['rows']

        # Use tqdm to display progress bar
        for table_row in tqdm(rows, desc=f"Fetching data for {exchange.upper()}"):
            csv_row = {table_headers.get(key, None): value for key, value in table_row.items()}
            combined_data.append(csv_row)

    # Write the combined data to the CSV file
    with open('Stocks.csv', 'w', newline='') as f_output:
        csv_output = csv.DictWriter(f_output, fieldnames=table_headers.values(), extrasaction='ignore')
        csv_output.writeheader()
        csv_output.writerows(combined_data)

    # Update stock_data with symbols from the combined data
    df = pd.read_csv('Stocks.csv')
    stock_data = df['Symbol'].tolist()


def load_reddit_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def analyze_sentiment(text, analyzer):
    return analyzer.polarity_scores(text)

def reddit_data(info, data_type):
    all_data = {}
    subreddit_file_path = f"subreddits/{data_type}_subreddits.txt"
    subreddits = pd.read_csv(subreddit_file_path, header=None)
    subreddit_list = subreddits[0].tolist()
    post_limit = 20

    for sub in subreddit_list:
        print(f"Processing subreddit: {sub}")
        subreddit_data = []

        try: 
            for submission in info.subreddit(sub).hot(limit=post_limit):
                if submission.upvote_ratio >= 0.65 and submission.ups >= 20:
                    post_data = {
                        'title': submission.title,
                        'id': submission.id,
                        'author': str(submission.author),
                        'created_utc': submission.created_utc,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'url': submission.url,
                        'comments': []
                    }
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if comment.score >= 3:
                            comment_data = {
                                'author': str(comment.author),
                                'score': comment.score,
                                'body': comment.body,
                                'created_utc': comment.created_utc
                            }
                            post_data['comments'].append(comment_data)
                    subreddit_data.append(post_data)
        except prawcore.exceptions.Forbidden:
                print(f"Skipping subreddit '{sub}' due to forbidden error.")
                continue
        all_data[sub] = subreddit_data

    with open('reddit_data.json', 'w') as json_file:
        json.dump(all_data, json_file, indent=4)

    print(f"Data saved to reddit_data.json")

def process_reddit_data(data):
    analyzer = SentimentIntensityAnalyzer()
    results = {}

    for subreddit, posts in data.items():
        print(f"Analyzing subreddit: {subreddit}")
        subreddit_results = []
        
        for post in posts:
            post_sentiment = analyze_sentiment(post['title'], analyzer)
            post_data = {
                'title': post['title'],
                'sentiment': post_sentiment,
                'comments': []
            }
            
            for comment in post['comments']:
                comment_sentiment = analyze_sentiment(comment['body'], analyzer)
                comment_data = {
                    'body': comment['body'],
                    'sentiment': comment_sentiment
                }
                post_data['comments'].append(comment_data)
            
            subreddit_results.append(post_data)
        
        results[subreddit] = subreddit_results
    
    return results

def stock_analysis(data):
    stock_sentiments = {}

    for subreddit, posts in data.items():
        print(f"Analyzing stocks in subreddit: {subreddit}")
        for post in posts:
            for symbol in stock_data:
                if re.search(r'\b' + re.escape(str(symbol)) + r'\b', post['title']):
                    sentiment = post['sentiment']
                    if symbol not in stock_sentiments:
                        stock_sentiments[symbol] = {'compound': 0, 'count': 0}
                    stock_sentiments[symbol]['compound'] += sentiment['compound']
                    stock_sentiments[symbol]['count'] += 1

            for comment in post['comments']:
                for symbol in stock_data:
                    if re.search(r'\b' + re.escape(str(symbol)) + r'\b', post['title']):
                        sentiment = comment['sentiment']
                        if symbol not in stock_sentiments:
                            stock_sentiments[symbol] = {'compound': 0, 'count': 0}
                        stock_sentiments[symbol]['compound'] += sentiment['compound']
                        stock_sentiments[symbol]['count'] += 1

    for symbol, data in stock_sentiments.items():
        data['average_sentiment'] = data['compound'] / data['count']

    return stock_sentiments

def save_sentiment_results(results, filepath):
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Sentiment analysis results saved to {filepath}")

def upload_image_to_firebase(file_path, destination_blob_name):
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    blob.make_public() 
    print(f"Uploaded {file_path} to {destination_blob_name}, accessible at {blob.public_url}")
    return blob.public_url

def visualize(filepath):
    with open(filepath, 'r') as file:
        stock_sentiments = json.load(file)
    
    top_stocks = sorted(stock_sentiments.items(), key=lambda x: x[1]['count'], reverse=True)[:8]
    
    for index, stock in enumerate(top_stocks, start=0):
        symbol = stock[0]
        avg_sentiment = stock[1]['average_sentiment']  # expected to be between 0 and 1 now
        
        fig = go.Figure()

        # Build gradient steps from 0 to avg_sentiment
        steps = []
        num_steps = 200
        for i in range(num_steps):
            step_value = i * (1 / num_steps)  # goes from 0 to 1 in increments of 1/num_steps
            if step_value > avg_sentiment:
                break  
            # Use color_value based on the step position
            color_value = i / num_steps
            if color_value < 0.5:
                r = 255
                g = int(255 * (color_value * 2))
                b = 0
            else:
                r = int(255 * (2 - 2 * color_value))
                g = 255
                b = 0
            color = f'rgb({r}, {g}, {b})'
            steps.append({'range': [step_value, step_value + (1 / num_steps)], 'color': color})
        
        # Since all sentiments are >= 0, use yellow for 0 and green otherwise
        needle_color = 'yellow' if avg_sentiment == 0 else 'green'

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment,
            title={'text': f"{symbol}", 'font': {'size': 24, 'color': 'black'}},  
            number={'valueformat': '.1%', 'font': {'size': 24, 'color': 'black'}}, 
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': needle_color, 'thickness': 0},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': steps,
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 1,
                    'value': avg_sentiment
                }
            }
        ))

        fig.update_layout(
            template="plotly_dark",
            margin={'t': 20},  
            paper_bgcolor='rgba(0,0,0,0)',  
            plot_bgcolor='rgba(0,0,0,0)',  
            annotations=[],
            font=dict(size=18, color='black')  
        )

        local_filename = f"gauges/stock_sentiment_gauge_{index}.png"
        fig.write_image(local_filename, width=600, height=600)
        print(f"Saved {local_filename}")
        
        # Upload to Firebase Storage using the local file path
        destination_blob_name = f'gauges/stock_sentiment_gauge_{index}.png'
        public_url = upload_image_to_firebase(local_filename, destination_blob_name)

def main():
    crypto = False

    data_procurement()
    if crypto:
        data_type = 'crypto'
    else:
        data_type = 'stock'

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    username = os.getenv("USERNAME")
    user_agent = os.getenv("USER_AGENT")
    password = os.getenv("PASSWORD")

    login = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    username=username,
    user_agent=user_agent,
    password=password
    )
    
    reddit_data(login, data_type)
    data = load_reddit_data('reddit_data.json')
    sentiment_results = process_reddit_data(data)
    save_sentiment_results(sentiment_results, 'sentiment_results.json')
    stock_sentiments = stock_analysis(sentiment_results)
    save_sentiment_results(stock_sentiments, 'stock_sentiment_results.json')
    visualize("stock_sentiment_results.json")


if __name__ == '__main__':
    main()