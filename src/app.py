# %%
import pandas as pd
import dash
from dash import html as html
from dash import dcc
from dash import callback
from datetime import datetime as dt
from dash.dependencies import Output, Input, State
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np
import seaborn as sns
import plotly.express as px


yfin.pdr_override()


# %%
df = pd.read_csv("src/nasdaq.csv")

options = []

for company in df.index:
    options.append({"label": df["Name"][company],
                    "value": df["Symbol"][company]})

# %%
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
             html.H1("Stock Analysis Web App"),
             html.Div([
                 html.H2("Select one or more stocks:"),
                 dcc.Dropdown(
                     id = "graph-dropdown",
                     options = options,
                     multi = True
                 )
             ]),
             html.Div([
                 html.H2("Select Date:"),
                 dcc.DatePickerRange(
                     id = "datepicker",
                     min_date_allowed = dt(2015, 1, 1),
                     max_date_allowed = dt.now(),
                     start_date = dt(2021, 1, 1),
                     end_date = dt.now()
                 )
             ]),
             html.Div([
                 html.Button(
                     id = "submit-button-main",
                     n_clicks = 0,
                     children = "Submit"
                 )
             ]),
             html.Div([
                 dcc.Graph(
                     id = "stock-graph",
                 )
             ]),
             html.Div([
                dcc.Tabs(
                    id = "tabs",
                    value = "tab-1",
                    children = [
                        dcc.Tab(label = "Monte Carlo Simulation", value = "tab-1"),
                        dcc.Tab(label = "Correlation Comparison", value = "tab-2")
                    ]
                )
             ]),
            html.Div(id='tabs-content')

])



# %%
@callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H2('Monte Carlo Simulation'),
            html.Div([
                dcc.Markdown('''
#### Why is this important?

Monte Carlo simulations are a way to model probability outcomes in financial markets. They can help investors understand the risk and potential returns of their investment portfolios. They do this by running multiple simulations of a portfolio's performance, based on the historical performance and volatility of the assets in the portfolio. The simulations generate a range of potential outcomes, and these outcomes can be plotted on a Monte Carlo graph. You read the graph by looking at where the paths end up. If a large number of paths end up at high values, that's a good sign for your investment. However, if many paths end at lower values, that could indicate a riskier investment.''')
            ]),
            html.Div([
                dcc.Dropdown(
                    id = "montecarlo-dropdown",
                    placeholder = "Select a stock",
                    options = options
                )
            ]),
            html.Div([
                dcc.Input(
                    id = "input-days",
                    placeholder = "Days")
            ]),
            html.Div([
                dcc.Input(
                    id = "input-runs",
                    placeholder = "Runs")
            ]),
            html.Div([
                html.Button(
                     id = "submit-button-montecarlo",
                     n_clicks = 0,
                     children = "Submit"
                 )
            ]),
            html.Div([
                dcc.Graph(
                    id = "montecarlo-graph"
                )
            ])
        ])
        
    elif tab == 'tab-2':
        return html.Div([
            html.H2('Correlation Analysis'),
            html.Div([
                dcc.Markdown('''
#### Why is this important?

A diversified stock portfolio is important because it spreads investment risk across multiple assets and industries. By holding a variety of stocks, rather than focusing on a single company or sector, investors can reduce their exposure to individual stock fluctuations and potential losses. Diversification helps to safeguard against the impact of unforeseen events or poor performance of any single stock, increasing the chances of achieving more stable and consistent returns over time. ''')
            ]),
            html.Div([
                dcc.Dropdown(
                    id = "correlation-x-dropdown",
                    options = options,
                    placeholder = "X-axis stock"
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id = "correlation-y-dropdown",
                    options = options,
                    placeholder = "Y-axis stock"
                )
            ]),
            html.Div([
                html.Button(
                    id = "submit-button-correlation",
                    n_clicks = 0,
                    children = "Submit"
                )
            ]),
            html.Div([
                dcc.Graph(
                    id = "correlation-graph"
                )
            ])
        ])

# %%
@app.callback(Output("stock-graph", "figure"),
              [Input("submit-button-main", "n_clicks")],
              [State("graph-dropdown", "value"),
               State("datepicker", "start_date"),
               State("datepicker", "end_date")])

def update_main_graph(number_of_clicks, stocks, start_date, end_date):

    start = dt.strptime(start_date[:10], "%Y-%m-%d")

    end = dt.strptime(end_date[:10], "%Y-%m-%d")

    data = []

    for stock in stocks:
        stock_df = pdr.get_data_yahoo(stock, start, end)

        dates = []

        for row in range(len(stock_df)):
            new_date = str(stock_df.index[row])

            new_date = new_date[0:10]

            dates.append(new_date)

        stock_df["Date"] = dates
        
        data.append({
            "x": stock_df["Date"],
            "y": stock_df["Adj Close"],
            "name": stock
        })

    figure = {
        "data": data,
        "layout": {"title": "Stock Data"}
    }

    return figure

# %%
def get_current_stock_price(symbol):
    ticker = yfin.Ticker(symbol)
    historical_data = ticker.history(period="1d")
    current_price = historical_data['Close'][-1]
    return current_price

# %%
@app.callback(Output("montecarlo-graph", "figure"),
              [Input("submit-button-montecarlo", "n_clicks")],
              [State("montecarlo-dropdown", "value"),
               State("input-days", "value"),
               State("input-runs", "value")])
def update_montecarlo_graph(number_of_clicks, stock, days, runs):

    # Convert `days` and `runs` to integers
    days = int(days)
    runs = int(runs)

    # Fix the datetime syntax
    start = dt(dt.now().year - 1, dt.now().month, dt.now().day)
    end = dt(dt.now().year, dt.now().month, dt.now().day)

    adjusted_closing_dataframe = pdr.get_data_yahoo(stock, start, end)["Adj Close"]
    returns = adjusted_closing_dataframe.pct_change().dropna()

    starting_price = get_current_stock_price(stock)
    
    dt_value = 1/days

    sigma = returns.std()
    mu = returns.mean()

    # Initialize `prices` as a 2D array
    prices = np.zeros((runs, days))
    prices[:, 0] = starting_price

    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run the simulation `runs` number of times
    for run in range(runs):
        for day in range(1, days):
            shock[day] = np.random.normal(loc=mu*dt_value, scale=sigma*np.sqrt(dt_value))
            drift[day] = mu * dt_value
            prices[run, day] = prices[run, day-1] + (prices[run, day-1] * (drift[day] + shock[day]))
    
    # Create a list of traces
    traces = []
    for run in range(runs):
        traces.append({
            "x": list(range(days)),
            "y": prices[run, :].tolist(),
            "mode": "lines"
        })

    # Create the figure
    figure = {
        "data": traces,
        "layout": {"title": "Monte Carlo Simulation"}
    }

    return figure


# %%
@app.callback(Output("correlation-graph", "figure"),
              [Input("submit-button-correlation", "n_clicks")],
              [State("correlation-x-dropdown", "value"),
               State("correlation-y-dropdown", "value")])
def update_seaborn_plot(number_of_clicks, stock_x, stock_y):
    start = dt(dt.now().year - 1, dt.now().month, dt.now().day)
    end = dt(dt.now().year, dt.now().month, dt.now().day)
     
    stocks = [stock_x, stock_y]
    adjusted_closing_dataframe = pdr.get_data_yahoo(stocks, start, end)["Adj Close"]
    percent_change = adjusted_closing_dataframe.pct_change()

    # Drop missing values before plotting
    percent_change = percent_change.dropna()

    # Calculate the correlation
    correlation = percent_change.corr().iloc[0, 1]

    # Create a scatter plot using Plotly Express
    fig = px.scatter(percent_change, x=stock_x, y=stock_y)

    # Add an annotation for the correlation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.2%}",
        showarrow=False,
        font=dict(size=14),
    )

    return fig

# %%
app.run()


