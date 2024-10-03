import requests

## Plot stock price (data from file)
with open("D:/sushil/expedite_commerce/stock_data.csv", "rb") as file:
    files = {'chatFile': file}
    response = requests.post('http://localhost:8004/plot', files=files)

## Analyze and identify top performing stocks (data from Yahoo API)
with open("D:/sushil/expedite_commerce/stock_data.csv", "rb") as file:
    files = {'chatFile': file}
    response = requests.post('http://localhost:8004/analyze', files=files)
  
## Analyze and identify top performing stocks (data from file)
stocks = ['bajfinance.ns', 'reliance.ns', 'hindalco.ns', 'zomato.ns', 'tatasteel.ns', 'yesbank.ns', 'idfc.ns', 'vedl.ns', 'iex.ns', 'infy.ns']    
response = requests.post('http://localhost:8004/analyze_yahoo_api', json={'stocks':stocks})

## Plot stock price (data from Yahoo API)
stocks = ['bajfinance.ns', 'reliance.ns', 'hindalco.ns', 'zomato.ns', 'tatasteel.ns', 'yesbank.ns', 'idfc.ns', 'vedl.ns', 'iex.ns', 'infy.ns']    
response = requests.post('http://localhost:8004/plot_yahoo_api', json={'stocks':stocks})
