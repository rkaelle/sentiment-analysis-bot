import requests
import pandas as pd
import json

# Twelve Data API key
api_key = "44a3c5a94e7142e39ce0dadc6c2f914f"
base_url = "https://api.twelvedata.com/stocks"
marketcap_url = "https://api.twelvedata.com/quote"

# Parameters for the API call
params = {
    "apikey": api_key,
    "exchange": "NYSE"

}

# API request
response = requests.get(marketcap_url, params=params)


data = response.json()

with open('market_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# Convert data to a dataframe
stocks_df = pd.DataFrame(data)
    
    # Display the top 1000 stocks
print(stocks_df.head(1000))
