import dash
from dash import dcc, html
from dash.dependencies import Input, Output, ALL
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import json
from dash import callback_context


# Function to preprocess data
def preprocess_data():
    df = pd.read_csv("full_processed_data_sorted.csv")
    selected_features = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Interest_Rate"]
    selected_columns = selected_features + ["Num_Credit_Card", "Credit_Utilization_Ratio",
                                            "Credit_History_Age", "Num_of_Delayed_Payment"]
    selected_df = df[selected_columns].copy()
    selected_df = selected_df[selected_df['Annual_Income'] < 150000]
    credit_score_column = df["Credit_Score"]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(selected_df[selected_features])
    normalized_df = pd.DataFrame(normalized_data, columns=selected_features)

    normalized_df = pd.concat([normalized_df, credit_score_column], axis=1)
    normalized_df['Credit_Score'] = pd.Categorical(normalized_df['Credit_Score'], categories=['Poor', 'Standard', 'Good'], ordered=True)
    mean_values = normalized_df.groupby('Credit_Score').mean()

    return selected_features, selected_df, mean_values, scaler


selected_features, selected_df, mean_values, scaler = preprocess_data()


# Function to create slider configurations
def create_slider_config(column_name, df, min_val=None, max_val=None, step=None, marks=None, label_generator=None):
    if min_val is None: min_val = int(df[column_name].min())
    if max_val is None: max_val = int(df[column_name].max())
    if step is None: step = 1
    if marks is None:
        if label_generator:
            marks = {i: label_generator(i) for i in range(min_val, max_val + 1, step)}
        else:
            marks = {i: str(i) for i in range(min_val, max_val + 1, step)}

    return {
        'id': {'type': 'slider', 'index': column_name},
        'min': min_val,
        'max': max_val,
        'step': step,
        'value': df[column_name].mean(),
        'marks': marks
    }


# Custom label generator for 'Interest_Rate'
def interest_rate_label_generator(value):
    return f"{value}-{value + 2}"


# Slider configurations
column_config = {
    'Age': create_slider_config('Age', selected_df, min_val=15, max_val=100, step=1, marks={i: str(i) for i in range(15, 101, 5)}),
    'Annual_Income': create_slider_config('Annual_Income', selected_df, min_val=0, max_val=500000, step=10000, marks={i: f"${i}" for i in range(0, 500001, 50000)}),
    'Monthly_Inhand_Salary': create_slider_config('Monthly_Inhand_Salary', selected_df, min_val=0, max_val=20000, step=100, marks={i: f"${i}" for i in range(0, 20001, 1000)}),
    'Num_Bank_Accounts': create_slider_config('Num_Bank_Accounts', selected_df),
    'Interest_Rate': create_slider_config('Interest_Rate', selected_df, min_val=0, max_val=40, step=2, label_generator=interest_rate_label_generator)
}

app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': col, 'value': col} for col in selected_df.columns],
            multi=True),
        html.Div(id='slider-container', style={'margin-bottom': '20px'}),
        dcc.Graph(id='selected-values-graph', style={'display': 'none'}),
        html.P(id='selected-values-paragraph')
    ], style={'width': '50%', 'text-align': 'center'}),
    html.Div(id='sliders-output-container', style={'margin-top': '20px'}),  # Div for displaying slider values
    html.Div(id='figure-output-container', style={'margin-top': '20px'})  # Div for displaying the figure or message
])


# Callback to update slider components
@app.callback(
    [Output('slider-container', 'children'),
     Output('sliders-output-container', 'children')],
    [Input('column-dropdown', 'value')]
)
def update_sliders(selected_columns):
    if not selected_columns:
        return 'Please select at least one column.', None

    sliders = []
    slider_values_divs = []

    for col in selected_columns:
        config = column_config.get(col)
        if config:
            slider = dcc.Slider(**config)
            sliders.append(html.Div([slider], style={'margin-bottom': '20px'}))
            slider_values_divs.append(html.Div(id={'type': 'slider-output', 'index': col}))

    return sliders, slider_values_divs


@app.callback(
    [Output({'type': 'slider-output', 'index': ALL}, 'children'),
     Output('figure-output-container', 'children')],
    [Input({'type': 'slider', 'index': ALL}, 'value')]
)
def update_slider_value_display(values):
    ctx = callback_context

    results = []
    new_user_data = {}

    if not ctx.triggered:
        return results, html.Div('No data available.')

    # Iterate over each input value and its corresponding ID
    for value, input_id in zip(values, ctx.inputs_list[0]):
        # Access column name directly from the input ID dictionary
        col = input_id['id']['index']
        new_user_data[col] = value
        results.append(f"{col} Current value: {int(value)}")

    print(results)

    if len(new_user_data) >= 3:

        remove_later = []
        for item in selected_features:
            if item not in new_user_data:
                new_user_data[item] = 0
                remove_later.append(item)

        new_user_df = pd.DataFrame([new_user_data])

        scaled_new_user_data = scaler.transform(new_user_df[selected_features])
        scaled_new_user_df = pd.DataFrame(scaled_new_user_data, columns=selected_features)

        df_combined = pd.concat([scaled_new_user_df, mean_values.loc['Good'].to_frame().T], ignore_index=True)
        max_value = df_combined.values.max() + 0.1
        fig = go.Figure()

        df_combined = df_combined.drop(columns=remove_later)

        for i, row_values in enumerate(df_combined.values):
            fig.add_trace(go.Scatterpolar(
                r=list(row_values) + [row_values[0]],  # Repeat the first point after the final point
                theta=list(df_combined.columns) + [df_combined.columns[0]],  # Repeat the first theta after the final theta
                name=f'Data {i}',
                mode='lines+markers',  # Add lines to connect the points
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )
            ),
            showlegend=True,
            title='Solar Graph',
        )
        return results, dcc.Graph(figure=fig)

    return results, html.Div('Please select at least three columns.', style={'display': 'block'})


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
