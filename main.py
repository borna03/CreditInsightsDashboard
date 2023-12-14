import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import random
import plotly.graph_objects as go

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
        html.P("Salary influences credit scores by impacting an individual's ability to manage and repay debts. A higher salary often correlates with increased financial stability, "
               "making it easier for individuals to meet their financial obligations, which positively affects their credit score. A strong credit score is crucial as it determines one's "
               "creditworthiness, impacting the ability to secure loans, favorable interest rates, and access to financial opportunities.", style={'text-align': 'center', 'color': '#555'}),
    ], style={'width': '45%', 'margin-left': '50%', 'text-align': 'center'}),  # Center and set width to 50%

    html.Div([
        html.Div([
            html.Div([
                html.P(
                    "Salary", style={'margin': '0','text-align': 'left'}),
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
                    "Nmr. Bank Accounts", style={'margin': '0','text-align': 'right'}),
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
                "To boost your credit score, it's essential to make tackling outstanding debts a top priority, working on reducing them gradually. Improve your financial habits by responsibly "
                "managing non-mortgage bank accounts, ensuring a steady and positive account history, which can positively impact your creditworthiness. Consider exploring opportunities for career "
                "growth or additional income streams, recognizing your current salary rating, to address financial concerns and strive for an overall improvement in your financial situation.",
                style={'text-align': 'center', 'color': '#555'}),
            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'Salary', 'value': 'scatter'},
                    {'label': 'Outstanding Dept', 'value': 'line'},
                    {'label': 'Nmr. Bank Accounts', 'value': 'funnel'}
                ],
                value=None,  # default selected option
                style={}),
            html.Div(id='selected-option-output'),  # Output to display selected option
            dcc.Graph(
                id='dynamic-graph',  # Use a new ID for the dynamic graph
                style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
            ),
        ], style={'width': '48%', 'text-align': 'center', 'margin-top': '20px'}),  # Set width to 48%
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

])


@app.callback(
    [Output('dynamic-graph', 'figure'),
     Output('dynamic-graph', 'style')],
    [Input('my-dropdown', 'value')]
)
def update_dynamic_graph(selected_option):
    if selected_option is None:
        return dash.no_update, {'display': 'none'}
    elif selected_option == 'scatter':
        return px.scatter(df, x='X', y='Y', title='Salary scatter plot'), {'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
    elif selected_option == 'line':
        return px.line(df, x='X', y='Y', title='Line Plot'), {'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
    elif selected_option == 'funnel':
        return px.funnel(df, x='Value', y='Category', title='Funnel Graph'), {'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}
    else:
        # Default to scatter plot and display the graph
        return px.scatter(df, x='X', y='Y', title='Scatter Plot'), {'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}


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
