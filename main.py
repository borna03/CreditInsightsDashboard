import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import random
import credit_score_related_with
import plotly.graph_objects as go

#### Loading and preping data for plotting
# Load the CSV file into a DataFrame
plot_discrete_data = pd.read_csv("full_processed_data_sorted.csv")

# Prepare the DataFrame for plotting continous data
# Create a copy of the DataFrame for plotting
plot_continuous_data = plot_discrete_data.copy()

# Convert necessary columns to numeric
columns_to_convert = ['Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_History_Age',
                      'Total_EMI_per_month', 'Amount_invested_monthly',
                      'Monthly_Balance', 'Credit_Utilization_Ratio', 'Annual_Income']

for col in columns_to_convert:
    plot_continuous_data[col] = pd.to_numeric(plot_continuous_data[col], errors='coerce')

# Map Credit_Score to numeric values
credit_score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
plot_continuous_data['Credit_Score_Numeric'] = plot_continuous_data['Credit_Score'].map(credit_score_mapping)

# Drop NaN values (or handle them as per your analysis requirement)
plot_continuous_data = plot_continuous_data.dropna(subset=columns_to_convert + ['Credit_Score_Numeric'])
####

categories = ['Occupation', 'Salary', 'Outstanding Dept', 'Nmr. Bank Accounts']
num_additional_values = 10

data = {
    'Category': ['Occupation', 'Salary', 'Outstanding Dept', 'Nmr. Bank Accounts'],
    'Value': [-5, 10, -15, 20]
}

df = pd.DataFrame({
    'Category': [category for _ in range(num_additional_values + 1) for category in categories],
    'Value': [random.randint(-30, 30) for _ in range(num_additional_values + 1) for _ in categories],
    'X': [random.uniform(1, 3) for _ in range(num_additional_values + 1) for _ in categories],
    'Y': [random.randint(15, 50) for _ in range(num_additional_values + 1) for _ in categories]
})

app = dash.Dash(__name__)

custom_font = {
    'font-family': 'Helvetica, sans-serif',
    'color': '#333'
}

counter = 0
app.layout = html.Div([
    html.H1("Dashboard", style={'text-align': 'center', **custom_font}),

    html.Div([
        dcc.Graph(
            id='diverging-bar-chart',
            figure=px.bar(data, x='Value', y='Category', orientation='h', title='Diverging Bar Chart'),
            style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),

        dcc.Graph(
            id='scatter-plot',
            style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),
    ]),

    html.Div([
        html.P(
            "Salary influences credit scores by impacting an individual's ability to manage and repay debts. A higher salary often correlates with increased financial stability, "
            "making it easier for individuals to meet their financial obligations, which positively affects their credit score. A strong credit score is crucial as it determines one's "
            "creditworthiness, impacting the ability to secure loans, favorable interest rates, and access to financial opportunities.",
            style={'text-align': 'center', 'color': '#555'}),
    ], style={'width': '45%', 'margin-left': '50%', 'text-align': 'center'}),  # Center and set width to 50%

    html.Div([
        html.Div([
            html.Div([
                html.P(
                    "Salary", style={'margin': '0', 'text-align': 'left'}),
                dcc.Slider(
                    id='slider-1',
                    min=0,
                    max=10,
                    step=1,
                    marks={i: str(i) for i in range(11)},
                    value=5
                ),
                html.P(
                    "Outstanding Dept", style={'margin': '0'}),
                dcc.Slider(
                    id='slider-2',
                    min=0,
                    max=10,
                    step=1,
                    marks={i: str(i) for i in range(11)},
                    value=5,
                ),
                html.P(
                    "Nmr. Bank Accounts", style={'margin': '0', 'text-align': 'right'}),
                dcc.Slider(
                    id='slider-3',
                    min=0,
                    max=10,
                    step=1,
                    marks={i: str(i) for i in range(11)},
                    value=5
                ),
            ], style={'width': '75%', 'margin': 'auto'}),
            dcc.Graph(
                id='slider-bar-chart',
                style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
            ),
        ], style={'width': '48%', 'text-align': 'center', 'margin-top': '20px'}),  # Set width to 48%

        html.Div([
            html.P(
                "To better undestand how credit score formation works, it's essential to understand how individual factors play a role in its formation.",
                style={'text-align': 'center', 'color': '#555'}),
            html.P(
                "In the dropdown menu below, you can pick certain factors that influence credit score, and receive insight on how are they related to it. Each of the variables are going to be related to average credit score, which ranges from 1 to 3: 1 - bad, 2 - standart, 3 - good.",
                "",
                style={'text-align': 'center', 'color': '#555'}),
            dcc.Markdown('### Average Credit Score by:'),
            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'Occupation', 'value': 'Occupation'},
                    {'label': 'Credit Mix', 'value': 'Credit_Mix'},
                    {'label': 'Payment Behaviour', 'value': 'Payment_Behaviour'},
                    {'label': 'Number of Bank Accounts', 'value': 'Num_Bank_Accounts'},
                    {'label': 'Number of Credit Cards', 'value': 'Num_Credit_Card'},
                    {'label': 'Interest Rate', 'value': 'Interest_Rate'},
                    {'label': 'Monthly Inhand Salary', 'value': 'Monthly_Inhand_Salary'},
                    {'label': 'Outstanding Debt', 'value': 'Outstanding_Debt'},
                    {'label': 'Credit History Age', 'value': 'Credit_History_Age'},
                    {'label': 'Total EMI per Month', 'value': 'Total_EMI_per_month'},
                    {'label': 'Amount Invested Monthly', 'value': 'Amount_invested_monthly'},
                    {'label': 'Monthly Balance', 'value': 'Monthly_Balance'},
                    {'label': 'Credit Utilization Ratio', 'value': 'Credit_Utilization_Ratio'},
                    {'label': 'Annual Income', 'value': 'Annual_Income'},
                ],
                value=None,  # default selected option
                style={}),
            html.Div(id='selected-option-output',
                     style={
                         'border': '1px solid #e1e1e1',
                         'border-radius': '8px',
                         'background-color': '#f9f9f9',
                         'padding': '20px',
                         'margin-top': '10px',
                         'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
                         'transition': 'transform 0.3s ease-in-out',
                         'transform': 'translateY(-3px)',
                         'cursor': 'default'
                     }),  # Output to display selected option
            dcc.Graph(
                id='dynamic-graph',  # Use a new ID for the dynamic graph
                style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
            ),
        ], style={'width': '48%', 'text-align': 'center', 'margin-top': '20px'}),  # Set width to 48%
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

])

# For each plot that gets displayed by the dropdown, we add a description, helping the user interpret the specific plot.
plot_descriptions_df = pd.read_csv('plot_descriptions.csv')
plot_descriptions_dict = plot_descriptions_df.set_index('Plot Name')['Description'].to_dict()


@app.callback(
    [Output('dynamic-graph', 'figure'),
     Output('dynamic-graph', 'style'),
     Output('selected-option-output', 'children')],
    [Input('my-dropdown', 'value')],
)
def update_dynamic_graph(selected_option):
    if selected_option is None:
        return dash.no_update, {'display': 'none'}, ""  # No selection, so no update
    else:
        title_selected_option = selected_option.replace('_', ' ')
        if selected_option in ['Occupation', 'Credit_Mix', 'Payment_Behaviour']:
            fig = credit_score_related_with.plot_bar_chart(plot_discrete_data, selected_option, 'lightblue',
                                                           f'Average Credit Score by {title_selected_option}')
        elif selected_option in ['Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate']:
            fig = credit_score_related_with.plot_line_chart(plot_discrete_data, selected_option, 'lightblue',
                                                            f'Average Credit Score by {title_selected_option}')
        else:
            fig = credit_score_related_with.plot_regression_line(plot_continuous_data, selected_option, 'lightblue',
                                                                 'Credit_Score_Numeric',
                                                                 f'Average Credit Score by {title_selected_option}')

        # Get the description for the selected option from the loaded dictionary
        description = plot_descriptions_dict.get(selected_option, "No description available.")
        description_html = html.Div([
            html.Div('🔍 Interpretation:', style={
                'font-size': '18px',
                'font-weight': 'bold',
                'color': '#333',
                'margin-bottom': '10px'
            }),
            html.Div(description, style={
                'font-size': '16px',
                'color': '#555'
            })
        ], style={
            'border': '1px solid #e1e1e1',
            'border-radius': '8px',
            'background-color': '#f9f9f9',
            'padding': '20px',
            'margin-top': '10px',
            'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
            'transition': 'transform 0.3s ease-in-out',
            'transform': 'translateY(-3px)',
            'cursor': 'default'
        })

        return fig, {'display': 'block'}, description_html


@app.callback(
    Output('slider-bar-chart', 'figure'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value'),
     Input('slider-3', 'value')]
)
def update_slider_bar_chart(value_1, value_2, value_3):
    slider_data = pd.DataFrame({
        'Category': ['Salary', 'Outstanding Dept', 'Nmr. Bank Accounts'],
        'Value': [value_1, value_2, value_3]
    })

    slider_bar_chart = px.bar(slider_data, x='Value', y='Category', orientation='h', title='Slider Bar Chart')
    return slider_bar_chart


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('diverging-bar-chart', 'clickData')]
)
def update_scatter_plot(click_data):
    if click_data is None:
        return px.scatter(title='Scatter Plot')
    else:
        clicked_category = click_data['points'][0]['y']
        filtered_data = df[df['Category'] == clicked_category]
        scatter_plot = px.scatter(filtered_data, x='X', y='Y', title=f'Scatter Plot for {clicked_category}')
        return scatter_plot


if __name__ == '__main__':
    app.run_server(debug=True)
