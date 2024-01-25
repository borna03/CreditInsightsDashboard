import dash
from dash import dcc, html
from dash.dependencies import Input, Output, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import random
import logging
import credit_score_related_with_helpers
from slider_helpers import preprocess_data, create_slider_config, generate_figure
from diverging_bar_chart_helpers import plot_correlation_chart, plot_continuous_distribution, \
    plot_discrete_distribution, \
    plot_categorical_distribution, plot_loan_distribution

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#### Loading and preping data for plotting ( Koko )
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

# Load your data and preprocess for the radar chart ( Borna )
selected_features, selected_df, mean_values, scaler = preprocess_data()

# Define slider configurations
column_config = {
    'Age': create_slider_config('Age', selected_df, min_val=15, max_val=100, step=1,
                                marks={i: str(i) for i in range(15, 101, 5)}),
    'Annual_Income': create_slider_config('Annual_Income', selected_df, min_val=0, max_val=500000, step=10000,
                                          marks={i: f"${i}" for i in range(0, 500001, 50000)}),
    'Monthly_Inhand_Salary': create_slider_config('Monthly_Inhand_Salary', selected_df, min_val=0, max_val=20000,
                                                  step=100, marks={i: f"${i}" for i in range(0, 20001, 1000)}),
    'Num_Bank_Accounts': create_slider_config('Num_Bank_Accounts', selected_df),
    'Interest_Rate': create_slider_config('Interest_Rate', selected_df, min_val=0, max_val=40, step=2)
}

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

app = dash.Dash(__name__)  # external_stylesheets=[dbc.themes.DARKLY]

custom_font = {
    'font-family': 'Helvetica, sans-serif',
    'color': '#333'
}

counter = 0

app.layout = html.Div([
    html.H1("Dashboard", style={'text-align': 'center', **custom_font}),

    # Row for Diverging Bar Chart and Interpretation Card
    html.Div([
        dcc.Graph(
            id='diverging-bar-chart',
            figure=plot_correlation_chart(plot_discrete_data),
            style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),
        html.Div(id='selected-option-output', style={
            'width': '33%',  # Reduced width to give more space to the chart
            'maxHeight': '400px',  # Set a max height to handle overflow
            'overflowY': 'auto',  # Enable vertical scroll
            'display': 'inline-block',
            'vertical-align': 'top',
            'border': '1px solid #e1e1e1',
            'border-radius': '8px',
            'background-color': '#f9f9f9',
            'padding': '20px',
            'margin-left': '2%',  # Add a little space between the chart and the card
            'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
            'transition': 'transform 0.3s ease-in-out',
            'transform': 'translateY(-3px)',
            'cursor': 'default'
        })
    ], style={'display': 'flex', 'justify-content': 'start', 'align-items': 'flex-start'}),

    # Second row with Distribution Chart below the Diverging Bar Chart and Dynamic Graph on the right
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),
        dcc.Graph(
            id='dynamic-graph',
            style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
        )
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'}),

    # Radar chart with sliders
    html.Div([
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': col, 'value': col} for col in selected_features],
            multi=True,
            value=selected_features
        ),
        html.Div(id='slider-container'),
        html.Div(id='sliders-output-container'),
        html.Div(id='figure-output-container')
    ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

], style={'margin': '10px'})

# For each plot that gets displayed by the dropdown, we add a description, helping the user interpret the specific plot.
plot_descriptions_df = pd.read_csv('plot_descriptions.csv')
plot_descriptions_dict = plot_descriptions_df.set_index('Plot Name')['Description'].to_dict()


@app.callback(
    [Output('dynamic-graph', 'figure'),
     Output('dynamic-graph', 'style'),
     Output('selected-option-output', 'children')],
    [  # Input('my-dropdown', 'value'),
        Input('diverging-bar-chart', 'clickData')],  # Added clickData as input
)
def update_dynamic_graph(click_data):  # selected_option
    ctx = dash.callback_context

    # Check which input was triggered
    if not ctx.triggered:
        # If no inputs were triggered, return no update
        return dash.no_update, {'display': 'none'}, ""
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if input_id == 'diverging-bar-chart' and click_data is not None:
            # Extract the category from the click data if there's no dropdown selection
            selected_option = click_data['points'][0]['y'].replace(' ', '_')  # Assuming that's the correct field
        else:
            return dash.no_update, {'display': 'none'}, ""

    title_selected_option = selected_option.replace('_', ' ')
    if selected_option in ['Type_of_Loan']:
        fig = credit_score_related_with_helpers.plot_bar_type_of_loan(plot_discrete_data, 'lightblue',
                                                                      f'Average Credit Score by Type of Loan')
    elif selected_option in ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']:
        fig = credit_score_related_with_helpers.plot_bar_chart(plot_discrete_data, selected_option, 'lightblue',
                                                               f'Average Credit Score by {title_selected_option}')
    elif selected_option in ['Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate', 'Delay_from_due_date',
                             'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
        fig = credit_score_related_with_helpers.plot_line_chart(plot_discrete_data, selected_option, 'lightblue',
                                                                f'Average Credit Score by {title_selected_option}')
    else:
        fig = credit_score_related_with_helpers.plot_regression_line(plot_continuous_data, selected_option, 'lightblue',
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
    Output('slider-container', 'children'),
    [Input('column-dropdown', 'value')]
)
def update_sliders(selected_columns):
    if not selected_columns:
        return html.Div()  # Return empty div if no columns are selected

    sliders = []
    for col in selected_columns:
        if col in column_config:
            sliders.append(html.Div(dcc.Slider(**column_config[col]), style={'margin-bottom': '20px'}))
    return html.Div(sliders)


@app.callback(
    Output('figure-output-container', 'children'),
    [Input({'type': 'slider', 'index': ALL}, 'value')],
    [Input('column-dropdown', 'value')]
)
def update_radar_chart(slider_values, selected_columns):
    if not slider_values or not selected_columns:
        return html.Div()  # Return empty div if no slider values or columns are provided

    # Construct a dict from column names to values
    new_user_data = {col: val for col, val in zip(selected_columns, slider_values)}

    # Generate the radar chart figure with the current slider values
    fig = generate_figure(new_user_data, selected_features, mean_values, scaler)
    return dcc.Graph(figure=fig)


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('diverging-bar-chart', 'clickData')]
)
def update_scatter_plot(click_data):

    if click_data is None:
        return px.scatter(title='Scatter Plot')
    else:
        clicked_category = click_data['points'][0]['y']
        logging.info(f"Clicked Category: {clicked_category}")
        if clicked_category in ['Type_of_Loan']:
            fig = plot_loan_distribution(plot_discrete_data)
        elif clicked_category in ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']:
            fig = plot_categorical_distribution(plot_discrete_data, clicked_category)
        elif clicked_category in ['Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate', 'Num_of_Loan']:
            fig = plot_discrete_distribution(plot_discrete_data, clicked_category)
        else:
            fig = plot_continuous_distribution(plot_continuous_data, clicked_category)

        return fig


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
