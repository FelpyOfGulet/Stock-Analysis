# Stock Analysis Web App

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description
This is a Dash web application for performing analysis on stocks. The application provides three main features:

Main Page: Allows users to select one or more stocks from a dropdown menu, choose a date range, and view a plot of the adjusted close prices for the selected stocks over the chosen date range.
Monte Carlo Simulation: Users can select a single stock, specify the number of days and runs for the simulation, and then view a plot of a Monte Carlo simulation based on this information.
Correlation Comparison: Users can select two stocks and view a scatter plot showing the correlation between the percentage changes in the adjusted closing prices of the two stocks.

## Installation
To install the necessary dependencies, you can use pip:

```bash
pip install dash pandas pandas_datareader yfinance numpy seaborn plotly
```
## Usage
To run the application, navigate to the directory containing the Web App.py file and run the following command:

```bash
python "Stock Analysis.py"
```
This will start a local server. You can view the application by navigating to the URL provided in the terminal output (usually http://127.0.0.1:8050/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.