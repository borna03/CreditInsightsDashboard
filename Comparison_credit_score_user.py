


dropdown_options = [{'label': str(customer_id), 'value': customer_id} for customer_id in customer_changes.index]
# Callback for the 'all-changes-chart' that updates based on clickData

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
            customdata = [customer_id] * len(customer_data['Month'])  # Add customdata attribute
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
        marker_color=['green' if x > 0 else 'red' for x in pct_changes.values]
    ))

    fig.update_layout(
        title=f'Percentage Changes in Other Variables for Customer {customer_id}',
        xaxis_title='Percentage Change (%)',
        yaxis_title='Variable'
    )

    return fig
