import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import logging
import credit_score_related_with_helpers
from slider_helpers import preprocess_data, create_slider_config, generate_figure
from diverging_bar_chart_helpers import plot_correlation_chart, plot_continuous_distribution, \
    plot_discrete_distribution, \
    plot_categorical_distribution, plot_loan_distribution
import dash_table

arr = []
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
selected_features, selected_df, mean_values, scaler, test = preprocess_data()

# Define slider configurations
column_config = {
    'Age': create_slider_config('Age', selected_df, min_val=15, max_val=100, step=1,
                                marks={i: str(i) for i in range(15, 101, 5)}),
    'Annual_Income': create_slider_config('Annual_Income', selected_df, min_val=0, max_val=500000, step=10000,
                                          marks={i: f"${i}" for i in range(0, 500001, 50000)}),
    'Monthly_Inhand_Salary': create_slider_config('Monthly_Inhand_Salary', selected_df, min_val=0, max_val=20000,
                                                  step=100, marks={i: f"${i}" for i in range(0, 20001, 1000)}),
    'Num_Bank_Accounts': create_slider_config('Num_Bank_Accounts', selected_df),
    'Interest_Rate': create_slider_config('Interest_Rate', selected_df, min_val=0, max_val=40, step=2),
    'Num_Credit_Card': create_slider_config('Num_Credit_Card', selected_df),
    'Credit_Utilization_Ratio': create_slider_config('Credit_Utilization_Ratio', selected_df, min_val=20, max_val=50, step=1, marks={i: str(i) for i in range(20, 50, 3)}),
    'Num_of_Delayed_Payment': create_slider_config('Num_of_Delayed_Payment', selected_df, step=1, max_val=50, marks={i: str(i) for i in range(0, 50, 5)})
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

unique_users = plot_discrete_data['Customer_ID'].unique()

copy_continuous_data = plot_continuous_data.copy()

# Assuming plot_continuous_data is your DataFrame

# Filter out the specific customer ID
filtered_data = plot_continuous_data[plot_continuous_data['Customer_ID'] != 'CUS_0xb226']

# Now, group by 'Customer_ID', count unique 'Credit_Score_Numeric' values, sort, and get top 10
customer_changes = filtered_data.groupby('Customer_ID')['Credit_Score_Numeric'] \
                                .nunique() \
                                .sort_values(ascending=False) \
                                .head(10)

dropdown_options = [{'label': str(customer_id), 'value': customer_id} for customer_id in customer_changes.index]


# Callback for the 'all-changes-chart' that updates based on clickData

# Callback for the 'all-changes-chart' graph
@app.callback(
    Output('all-changes-chart', 'figure'),
    [Input('all-changes-chart', 'clickData'),
     Input('customer-dropdown', 'value')]
)
def update_all_changes_chart(click_data, dropdown_value):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Determine which input (dropdown or chart click) triggered the callback
    selected_customer_id = dropdown_value if triggered_id == 'customer-dropdown' else click_data['points'][0]['customdata'] if click_data else dropdown_value

    fig = go.Figure()

    # Loop through the customer IDs from the index (assumed to be in customer_changes DataFrame)
    for customer_id in customer_changes.index:
        customer_data = plot_continuous_data[plot_continuous_data['Customer_ID'] == customer_id]

        line_opacity = 0.1 if customer_id != selected_customer_id else 1.0
        line_width = 2 if customer_id != selected_customer_id else 4

        fig.add_trace(go.Scatter(
            x=customer_data['Month'],
            y=customer_data['Credit_Score_Numeric'],
            mode='lines+markers',
            name=str(customer_id),
            opacity=line_opacity,
            line=dict(width=line_width),
            customdata=[customer_id] * len(customer_data['Month'])  # Add customdata attribute
        ))

    fig.update_layout(
        title='Change in Credit Score Over Time For the 5 most changed user',
        xaxis_title='Month',
        yaxis_title='Credit Score',
        clickmode='event+select'
    )

    return fig


# Callback for the 'divergent-bar-chart' graph
@app.callback(
    Output('divergent-bar-chart', 'figure'),
    [Input('all-changes-chart', 'clickData'),
     Input('customer-dropdown', 'value')]
)
def update_divergent_bar_chart(click_data, dropdown_value):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Determine which input (dropdown or chart click) triggered the callback
    if triggered_id == 'all-changes-chart' and click_data:
        customer_id = click_data['points'][0]['customdata']
    elif triggered_id == 'customer-dropdown':
        customer_id = dropdown_value
    else:
        raise PreventUpdate

    customer_data = plot_continuous_data[plot_continuous_data['Customer_ID'] == customer_id]

    columns_to_include = [
        'Monthly_Inhand_Salary',
        'Outstanding_Debt',
        'Credit_History_Age',
        'Total_EMI_per_month',
        'Amount_invested_monthly',
        'Monthly_Balance',
        'Credit_Utilization_Ratio',
        'Annual_Income',
        'Num_of_Loan',
        'Interest_Rate',
        'Num_Credit_Card'
    ]

    pct_changes = customer_data[columns_to_include].pct_change().mean() * 100

    fig = go.Figure(go.Bar(
        x=pct_changes.values,
        y=columns_to_include,
        orientation='h',
        marker_color=['#1F77B4' if x > 0 else '#FF7F0E' for x in pct_changes.values]
    ))

    fig.update_layout(
        title=f'Percentage Changes in Other Variables for Customer {customer_id}',
        xaxis_title='Percentage Change (%)',
        yaxis_title='Variable'
    )

    return fig


# Your existing layout components...

#


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
    ], style={'display': 'flex', 'justify-content': 'start', 'align-items': 'center', 'max-width': '1500px'}),

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
    ], id='scatter-plot-container'),

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
    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-bottom': '30px'}),

    html.Div([
        html.Div([
            dcc.Slider(
                id='quality-slider',
                min=0,
                max=2,
                step=1,  # Ensure slider moves in discrete steps
                marks={
                    0: 'Poor',
                    1: 'Standard',
                    2: 'Good'
                },
                value=1  # Default value set to 'Standard'
            )
        ], style={'width': '40%', 'padding-left': '30%'}),  # Apply the style here
        html.Div([
            html.Div(id='figure-output-container', style={'flex': '1', 'padding-right': '20px'}),  # Container for displaying the radar chart graph
            html.Div(id='table-output-container', style={'flex': '1', 'padding-left': '20px'})  # New container for displaying the table
        ], style={'display': 'flex', 'justify-content': 'space-around', 'align-items': 'center', 'width': '100%', 'margin-bottom': '30px'})
    ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),

    # # Added components for user-dropdown and credit-score-plot
    # dropdown_layout,

    # dcc.Dropdown(id='customer-dropdown', options=dropdown_options, value=dropdown_options[0]['value']),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='customer-dropdown',
                options=dropdown_options,
                value=dropdown_options[0]['value'],  # Set default value to the first customer ID
                style={'width': '80%', 'margin': 'auto'}  # Set the width of the dropdown to 100%
            ),
            dcc.Graph(
                id='all-changes-chart',
                style={'height': '40vh', 'width': '75%'}  # Set both the height and width of the graph
            ),
            dcc.Graph(
                id='divergent-bar-chart',
                style={'height': '40vh', 'width': '75%'}  # Set both the height and width of the graph
            ),

        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'width': '100%'})
    ], style={'width': '100%'})

], style={'margin': '10px'})

# For each plot that gets displayed by the dropdown, we add a description, helping the user interpret the specific plot.
plot_descriptions_df = pd.read_csv('plot_descriptions.csv')
plot_descriptions_dict = plot_descriptions_df.set_index('Plot Name')['Description'].to_dict()
x_axis_dict = plot_descriptions_df.set_index('Plot Name')['x_axis'].to_dict()


@app.callback(
    [Output('dynamic-graph', 'figure'),
     Output('dynamic-graph', 'style'),
     Output('selected-option-output', 'children')],
    [  # Input('my-dropdown', 'value'),
        Input('diverging-bar-chart', 'clickData')],  # Added clickData as input
)
def update_dynamic_graph(click_data):  # selected_option
    ctx = dash.callback_context
    plot_discrete_data_copy1 = plot_discrete_data.copy()

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

    # Extract description from previously loaded csv
    description = plot_descriptions_dict.get(selected_option, "No description available.")

    # Extract x-axis title
    x_axis_title = x_axis_dict.get(selected_option, "No title available.")

    title_selected_option = selected_option.replace('_', ' ')
    if selected_option in ['Type_of_Loan']:
        fig = credit_score_related_with_helpers.plot_bar_type_of_loan(plot_discrete_data_copy1, 'lightblue',
                                                                      f'Average Credit Score by Type of Loan', x_axis_title)
    elif selected_option in ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']:
        fig = credit_score_related_with_helpers.plot_bar_chart(plot_discrete_data, selected_option, 'lightblue',
                                                               f'Average Credit Score by {x_axis_title}', x_axis_title)
    elif selected_option in ['Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate', 'Delay_from_due_date',
                             'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
        fig = credit_score_related_with_helpers.plot_line_chart(plot_discrete_data, selected_option, 'lightblue',
                                                                f'Average Credit Score by {x_axis_title}', x_axis_title)
    else:
        fig = credit_score_related_with_helpers.plot_regression_line(plot_continuous_data, selected_option, 'lightblue',
                                                                     'Credit_Score_Numeric',
                                                                     f'Average Credit Score by {x_axis_title}', x_axis_title)

    # Title for the markdown with an emoji
    title_with_emoji = f"📊 Key Insights on **{title_selected_option}**\n\n"

    # Get the description for the selected option from the loaded dictionary
    # description = plot_descriptions_dict.get(selected_option, "No description available.")

    # Full markdown text including the title
    full_markdown_text = title_with_emoji + description
    # description_html = html.Div([
    #     html.Div('🔍 Interpretation:', style={
    #         'font-size': '18px',
    #         'font-weight': 'bold',
    #         'color': '#333',
    #         'margin-bottom': '10px'
    #     }),
    #     html.Div(description, style={
    #         'font-size': '16px',
    #         'color': '#555'
    #     })
    # ], style={
    #     'border': '1px solid #e1e1e1',
    #     'border-radius': '8px',
    #     'background-color': '#f9f9f9',
    #     'padding': '20px',
    #     'margin-top': '10px',
    #     'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
    #     'transition': 'transform 0.3s ease-in-out',
    #     'transform': 'translateY(-3px)',
    #     'cursor': 'default'
    # })
    description_markdown = dcc.Markdown(children=full_markdown_text, style={
        'font-size': '18px',
        'color': '#555',
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

    return fig, {'display': 'block'}, description_markdown


@app.callback(
    Output('slider-container', 'children'),
    [Input('column-dropdown', 'value'),
     Input({'type': 'slider', 'index': ALL}, 'value')]
)
def update_sliders(selected_columns, slider_values):
    if not selected_columns:
        return html.Div()  # Return empty div if no columns are selected

    if not slider_values or len(slider_values) != len(selected_columns):
        # Set default values if slider values are not set
        slider_values = [column_config[col]['value'] for col in selected_columns]
    i = 0
    bella = 0
    sliders = []
    for col, val in zip(selected_columns, slider_values):
        if col in column_config:
            # Create a label for each slider with its name and current value
            slider_label = html.Label(f'{col.replace("_", " ").title()}: {round(val)}',
                                      style={'fontFamily': 'Helvetica, sans-serif',
                                             'margin-top': '20px',
                                             'color': 'rgb(51, 51, 51)'})

            # Create the slider excluding 'value' from column_config to avoid duplicate 'value' error
            slider_config = {key: column_config[col][key] for key in column_config[col] if key != 'value'}
            slider = dcc.Slider(**slider_config, value=val)

            # Add the label and the slider to the layout
            if i == 0:
                sliders.append(html.Div([slider_label, slider], style={'margin-bottom': '5px', 'margin-top': '25px'}))
                i += 1
            else:
                sliders.append(html.Div([slider_label, slider], style={'margin-bottom': '5px'}))

    return html.Div(sliders)


def std_dev_of_upper_quartile(df, column_name):
    # Calculate the 75th percentile (third quartile)
    third_quartile = df[column_name].quantile(0.75)

    # Filter the DataFrame
    filtered_df = df[df[column_name] >= third_quartile]

    # Calculate the standard deviation of the filtered DataFrame
    std_dev = filtered_df[column_name].std()

    return std_dev


def find_similar(attributes_selected, df, vals, score):
    stds = []
    for item in attributes_selected:
        stds.append([item, std_dev_of_upper_quartile(df, item)])

    for index, row in df.iterrows():
        print('x')
        counter = len(attributes_selected)
        if row['Credit_Score'] != score:
            pass
        else:
            for item in stds:
                difference = item[1]
                if item[0] == "Annual_Income":
                    # print(row[item[0]], vals[item[0]], difference, item)
                    if abs(row[item[0]] - vals[item[0]]) < 20000:
                        counter -= 1
                else:
                    # print(row[item[0]], vals[item[0]], difference, item)
                    if abs(row[item[0]] - vals[item[0]]) < difference:
                        counter -= 1
            if counter == 0:
                return row
    return None


@app.callback(
    [
        Output('figure-output-container', 'children'),  # Output for the radar chart
        Output('table-output-container', 'children')    # Output for the table
    ],
    [
        Input({'type': 'slider', 'index': ALL}, 'value'),  # Inputs from dynamically generated sliders
        Input('column-dropdown', 'value'),  # Input from the dropdown
        Input('quality-slider', 'value')  # Input from the quality slider
    ]
)
def update_radar_chart(slider_values, selected_columns, quality_value):
    global arr

    quality_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}

    if not slider_values or not selected_columns:
        return html.Div(), None  # Return empty div for figure and None for table if no input is provided

    # Construct a dict from column names to values
    new_user_data = {col: val for col, val in zip(selected_columns, slider_values)}
    arr = new_user_data

    # Generate the radar chart figure with the current slider values
    fig = generate_figure(new_user_data, selected_features, mean_values, scaler, quality_labels[quality_value])

    # Find a similar user and generate the table if applicable
    similar_user = find_similar(selected_columns, plot_discrete_data, new_user_data, quality_labels[quality_value])
    if similar_user is not None:
        table_data = [
            {'Feature': col, 'Value': similar_user[col]} for col in selected_columns + ["Occupation", "Credit_Score"]
        ]

        table = dash_table.DataTable(
            data=table_data,
            columns=[{'name': 'Similar user', 'id': 'Feature'}, {'name': 'Value', 'id': 'Value'}],
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            }
        )
    else:
        table = None

    # Return the figure and table as separate components
    return dcc.Graph(figure=fig), table


@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('scatter-plot-container', 'style')],
    [Input('diverging-bar-chart', 'clickData')]
)
def update_scatter_plot(click_data):
    plot_discrete_data_copy = plot_discrete_data.copy()
    style = {'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px', 'max-width': '1500px', 'padding-bottom': '35px'}
    if click_data is None:
        return dash.no_update, {'display': 'none'}
    else:
        clicked_category = click_data['points'][0]['y']

        # Extract x-axis title
        x_axis_title = x_axis_dict.get(clicked_category, "No title available.")

        logging.info(f"Clicked Category: {clicked_category}")
        if clicked_category in ['Type_of_Loan']:
            fig = plot_loan_distribution(plot_discrete_data_copy)
        elif clicked_category in ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']:
            fig = plot_categorical_distribution(plot_discrete_data_copy, clicked_category, x_axis_title)
        elif clicked_category in ['Num_Credit_Card', 'Num_Bank_Accounts', 'Interest_Rate', 'Num_of_Loan']:
            fig = plot_discrete_distribution(plot_discrete_data_copy, clicked_category, x_axis_title)
        else:
            fig = plot_continuous_distribution(plot_continuous_data, clicked_category, x_axis_title)

        return fig, style


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
