
#counter = 0




#unique_users = plot_discrete_data['Customer_ID'].unique()
#copy_continuous_data = plot_continuous_data.copy()

customer_changes = plot_continuous_data.groupby('Customer_ID')['Credit_Score_Numeric'].nunique().sort_values(ascending=False).head(25)
dropdown_options = [{'label': id, 'value': id} for id in customer_changes.index]
# Callback to update the graph
@app.callback(
    Output('line-chart', 'figure'),
    Input('customer-dropdown', 'value')
)
def update_line_chart(customer_id):
    # Filter the DataFrame for the selected customer
    customer_data = plot_continuous_data[plot_continuous_data['Customer_ID'] == customer_id]

    # Create the line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=customer_data['Month'], y=customer_data['Credit_Score_Numeric'],
                             mode='lines+markers', name='Credit Score', customdata=[customer_id]*len(customer_data)))
    fig.update_layout(title='Change in Credit Score Over Time for Customer {}'.format(customer_id),
                      xaxis_title='Month', yaxis_title='Credit Score')

    return fig

# Callback to update the all changes chart
@app.callback(
    Output('all-changes-chart', 'figure'),
    Input('all-changes-chart', 'id')
)
def update_all_changes_chart(id):
    # Create the all changes chart
    fig = go.Figure()
    for customer_id in customer_changes.index:
        customer_data = plot_continuous_data[plot_continuous_data['Customer_ID'] == customer_id]
        fig.add_trace(go.Scatter(x=customer_data['Month'], y=customer_data['Credit_Score_Numeric'],
                                 mode='lines+markers', name=customer_id, customdata=[customer_id]*len(customer_data)))

    fig.update_layout(title='Change in Credit Score Over Time for Top 25 Customers',
                      xaxis_title='Month', yaxis_title='Credit Score')

    return fig

# Callback to update the divergent bar chart
# Callback to update the divergent bar chart
# Callback to update the divergent bar chart
# Callback to update the divergent bar chart
@app.callback(
    Output('divergent-bar-chart', 'figure'),
    Input('customer-dropdown', 'value')
)
def update_divergent_bar_chart(customer_id):
    # Filter the DataFrame for the selected customer
    customer_data = plot_continuous_data[plot_continuous_data['Customer_ID'] == customer_id]

    # Specify the columns to include
    columns_to_include = ['Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_History_Age',
                          'Total_EMI_per_month', 'Amount_invested_monthly',
                          'Monthly_Balance', 'Credit_Utilization_Ratio', 'Annual_Income',
                          'Num_of_Loan','Interest_Rate','Num_Credit_Card']

    # Calculate the percentage changes in the specified variables
    pct_changes = customer_data[columns_to_include].pct_change().mean() * 100

    # Create the divergent bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pct_changes.values, y=pct_changes.index, orientation='h',
                         marker_color=['green' if x > 0 else 'red' for x in pct_changes.values]))
    fig.update_layout(title='Percentage Changes in Other Variables for Customer {}'.format(customer_id),
                      xaxis_title='Percentage Change (%)', yaxis_title='Variable')

    return fig






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

    # # Added components for user-dropdown and credit-score-plot
    # dropdown_layout,

    dcc.Dropdown(id='customer-dropdown', options=dropdown_options, value=dropdown_options[0]['value']),
    dcc.Graph(id='line-chart', clickData={'points': [{'customdata': dropdown_options[0]['value']}]}, style={'height': '50vh'}),
    dcc.Graph(id='all-changes-chart', style={'height': '50vh'}),
    dcc.Graph(id='divergent-bar-chart', style={'height': '50vh'})


], style={'margin': '10px'})