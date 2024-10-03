import yfinance as yf
import talib
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import json
import io
from io import BytesIO
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This allows requests from any origin. You can adjust this as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {'csv'}

prompt_temp = """Identify the top 3 stocks based on the given stock name and its indicators. Generate the output without any additional text for any explanation:
    {}
    [Top 3 stocks]:"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_stock_data(ticker, period='3mo', interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    data = data[['Date','Open', 'High', 'Low', 'Close', 'Volume']]
    data['stock_name'] = ticker
    return data

def calculate_metric(df):
    tp = len(df)
    SMA = talib.SMA(df['Close'], timeperiod=tp).iloc[-1]
    EMA = talib.EMA(df['Close'], timeperiod=tp).iloc[-1]
    RSI = talib.RSI(df['Close'], timeperiod=tp-1).iloc[-1]
    CCI = talib.CCI(df['High'],df['Low'],df['Close'], timeperiod=tp).iloc[-1]
    percentage_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0])/df['Close'].iloc[0])*100
    high = max(df['High'])
    low = min(df['Low'])
    return {'High_3month':high, 'Low_3month':low,'SMA':SMA, 'EMA':EMA, 'RSI':RSI, 'CCI':CCI , 'percentage_change':percentage_change}

def llm(prompt):
    client = Groq(
        api_key="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")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                }
            ],
        model="mixtral-8x7b-32768",
        temperature = 0.01,
    )
    
    return chat_completion.choices[0].message.content

@app.post('/plot')
async def plot_stock_prices(chatFile: UploadFile = File(...)): 
    if allowed_file(chatFile.filename):
        content = await chatFile.read()
        data = pd.read_csv(BytesIO(content))
        # Convert date column to datetime format
        data['Date'] = pd.to_datetime(data['Date']) 
        base64_list = []
        for stock_name, stock_df in data.groupby('stock_name'): 
            plt.plot(stock_df['Date'], stock_df['Close'], label=stock_name)
            plt.title('Stock Prices Over 3-Month Period')
            plt.xlabel('Date')
            plt.ylabel('Stock Price (Close)')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            base64_list.append(plot_base64)
        return {'plot':base64_list}
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
        
@app.post('/plot_yahoo_api')
async def plot_stock_prices_yahoo(request: Request): 
    request_dict = await request.json()
    stocks = request_dict.pop("stocks")
    df_list = []
    for stock in stocks:
        df_list.append(fetch_stock_data(stock))   
    data = pd.concat(df_list)
    # Convert date column to datetime format
    data['Date'] = pd.to_datetime(data['Date']) 
    base64_list = []
    for stock_name, stock_df in data.groupby('stock_name'): 
        plt.plot(stock_df['Date'], stock_df['Close'], label=stock_name)
        plt.title('Stock Prices Over 3-Month Period')
        plt.xlabel('Date')
        plt.ylabel('Stock Price (Close)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        base64_list.append(plot_base64)
    return {'plot':base64_list}
    

@app.post('/analyze')
async def analyze_stock(chatFile: UploadFile = File(...)):
    if allowed_file(chatFile.filename):
        content = await chatFile.read()
        df = pd.read_csv(BytesIO(content))
        tmp = ''
        for stock_name, stock_df in df.groupby('stock_name'):
            tmp += stock_name + '= '
            tmp += json.dumps(calculate_metric(stock_df)) + '\n'
        prompt = prompt_temp.format(tmp)
        result = llm(prompt)
        return {'result':result}
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

@app.post('/analyze_yahoo_api')
async def analyze_stock_yahoo(request: Request):
    request_dict = await request.json()
    stocks = request_dict.pop("stocks")
    df_list = []
    for stock in stocks:
        df_list.append(fetch_stock_data(stock))   
    df = pd.concat(df_list)
    tmp = ''
    for stock_name, stock_df in df.groupby('stock_name'):
        tmp += stock_name + '= '
        tmp += json.dumps(calculate_metric(stock_df)) + '\n'
    prompt = prompt_temp.format(tmp)
    result = llm(prompt)
    return {'result':result}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8004)
    
    
    
    