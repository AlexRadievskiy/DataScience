import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_data():
    ticker = "TON11419-USD"
    data = yf.download(ticker, start="2021-01-01", end="2023-01-01", interval="1mo")
    return data

# def parse_data():
#     url = "https://finance.yahoo.com/quote/TON11419-USD/history?period1=1630022400&period2=1694908800&interval=1mo&filter=history&frequency=1mo&includeAdjustedClose=true"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#
#     table = soup.find('table', {'data-test': 'historical-prices'})
#     rows = table.find_all('tr')
#
#     data = []
#     for row in rows[1:]:
#         columns = row.find_all('td')
#         date = columns[0].text
#         close_price = columns[4].text
#         data.append((date, close_price))
#
#     df = pd.DataFrame(data, columns=['Date', 'Close Price'])
#     return df

def save_to_csv(df):
    df.to_csv('parsed_data.csv')

def plot_trend(df):
    df['Close'].plot()
    plt.title('Dynamics of TON Coin')
    plt.show()

def get_statistics(df):
    return df.describe()

def model_data(df):
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)
    df['Predicted'] = model.predict(X)

    df[['Close', 'Predicted']].plot()
    plt.title('Real vs Predicted Data')
    plt.show()

if __name__ == "__main__":
    data = fetch_data()
    save_to_csv(data)
    plot_trend(data)
    print(get_statistics(data))
    model_data(data)
