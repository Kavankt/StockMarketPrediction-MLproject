import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import yfinance as yf
import plotly.graph_objs as go
from prophet import Prophet

# Initialize Dash app
app = dash.Dash(__name__)

# External CSS stylesheets
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cyborg/bootstrap.min.css']

# Set external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Function to analyze stock data and provide recommendation
def analyze_stock_data(stock_data):
    # Calculate daily returns
    daily_returns = stock_data['Close'].pct_change()

    # Calculate average daily return
    avg_daily_return = daily_returns.mean()

    if avg_daily_return > 0:
        return "Buy"
    elif avg_daily_return < 0:
        return "Sell"
    else:
        return "Hold"

# Function to train model and make predictions for future stock prices
def train_and_predict(stock_data):
    # Prepare data for Prophet
    df = stock_data.reset_index()
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']

    # Initialize Prophet model
    model = Prophet()

    # Fit model to data
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Predict future prices for the next 30 days
    forecast = model.predict(future)

    return forecast

# Layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Stock Data Visualization and Prediction", className="text-center mb-4"),
        html.Div([
            dcc.Input(id="stock-input", type="text", placeholder="Enter stock symbol...", className="form-control"),
            html.Button("Fetch Data", id="fetch-button", n_clicks=0, className="btn btn-primary mt-3")
        ], className="container text-center"),
        html.Div(id="stock-details", className="container mt-4"),
        html.Div(id="current-price-output", className="container mt-2"),
        html.Div(id="output", className="container text-center mt-2"),
        dcc.Graph(id='stock-chart', className="container mt-2"),
        html.Div(id='recommendation-output', className="container mt-2 text-center "),
        html.Div(id='prediction-output', className="container mt-2"),
    ])
])

# Callback to fetch and display live stock data
@app.callback(
    [Output("output", "children"),
     Output("stock-chart", "figure"),
     Output('stock-details', 'children'),
     Output('current-price-output', 'children'),
     Output('recommendation-output', 'children'),
     Output('prediction-output', 'children')],
    [Input("fetch-button", "n_clicks")],
    [State("stock-input", "value")]
)
def fetch_and_visualize_stock_data(n_clicks, stock_symbol):
    if n_clicks > 0:
        if stock_symbol:
            try:
                # Fetch live stock data
                stock_data = yf.download(stock_symbol, period='6mo')
                if stock_data.empty:
                    return "No data available for this stock symbol.", {}, html.Div(""), html.Div(""), html.Div(""), html.Div("")

                # Get details of the company/stock
                stock_info = yf.Ticker(stock_symbol)
                company_info = stock_info.info
                stock_details = html.Div([
                    html.H3("Stock Details"),
                    html.P(f"Company Name: {company_info['longName']}"),
                    html.P(f"Symbol: {company_info['symbol']}"),
                    html.P(f"Sector: {company_info['sector']}"),
                    html.P(f"Industry: {company_info['industry']}")
                ])

                # Get current stock price
                current_price = stock_data['Close'].iloc[-1]
                current_price_text = html.Div(f"Current Stock Price: {current_price}", className="lead")

                # Visualize live stock data
                trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price')
                layout = go.Layout(title=f'{stock_symbol} Live Stock Prices', xaxis=dict(title='Time'), yaxis=dict(title='Price'))
                figure = go.Figure(data=[trace], layout=layout)

                # Analyze stock data and provide recommendation
                recommendation = analyze_stock_data(stock_data)
                recommendation_text = html.P(f"Recommendation: {recommendation}", className="lead text-primary")

                # Train model and make predictions
                predictions = train_and_predict(stock_data)
                prediction_text = html.Div([
                    html.H3("Predicted Stock Prices"),
                    dcc.Graph(
                        id='predicted-stock-chart',
                        figure={
                            'data': [
                                go.Scatter(x=predictions['ds'], y=predictions['yhat'], mode='lines', name='Predicted Price')
                            ],
                            'layout': {
                                'title': 'Predicted Stock Prices'
                            }
                        }
                    )
                ])

                return html.Div([
                    html.H4("Successfully fetched live data", className="text-center text-success"),
                    f"Successfully fetched live data for {stock_symbol}"
                ]), figure, stock_details, current_price_text, recommendation_text, prediction_text
            except Exception as e:
                return f"An error occurred: {str(e)}", {}, html.Div(""), html.Div(""), html.Div(""), html.Div("")
        else:
            return "Please enter a stock symbol.", {}, html.Div(""), html.Div(""), html.Div(""), html.Div("")
    else:
        return "", {}, html.Div(""), html.Div(""), html.Div(""), html.Div("")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
