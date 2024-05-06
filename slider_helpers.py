# helpers.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go


def preprocess_data():
    df = pd.read_csv("data/full_processed_data_sorted.csv")
    selected_features = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Interest_Rate", "Num_Credit_Card", "Credit_Utilization_Ratio",
                         "Num_of_Delayed_Payment"]  # 'Credit_History_Age', 'Monthly_Inhand_Salary',
    selected_columns = selected_features + ["Credit_History_Age"]

    selected_df = df[selected_columns].copy()
    selected_df = selected_df[selected_df['Annual_Income'] < 150000]
    credit_score_column = df["Credit_Score"]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(selected_df[selected_features])
    normalized_df = pd.DataFrame(normalized_data, columns=selected_features)

    normalized_df = pd.concat([normalized_df, credit_score_column], axis=1)
    normalized_df['Credit_Score'] = pd.Categorical(normalized_df['Credit_Score'], categories=['Poor', 'Standard', 'Good'], ordered=True)
    mean_values = normalized_df.groupby('Credit_Score').mean()

    return selected_features, selected_df, mean_values, scaler, selected_columns


def create_slider_config(column_name, df, min_val=None, max_val=None, step=None, marks=None, label_generator=None):
    if min_val is None:
        min_val = int(df[column_name].min())
    if max_val is None:
        max_val = int(df[column_name].max())
    if step is None:
        step = 1

    range_values = range(min_val, max_val + 1, step)
    if marks is None:
        if label_generator:
            marks = {i: label_generator(i) for i in range_values}
        else:
            marks = {i: str(i) for i in range_values}

    # Append '+' to the last label
    last_value = max(range_values)
    if label_generator:
        marks[last_value] = label_generator(last_value) + '+'
    else:
        marks[last_value] = str(last_value) + '+'

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


def generate_figure(new_user_data, selected_features, mean_values, scaler, lmao):
    print(lmao, "asdasdsa")
    remove_later = []
    for item in selected_features:
        if item not in new_user_data:
            new_user_data[item] = 0
            remove_later.append(item)

    new_user_df = pd.DataFrame([new_user_data])

    scaled_new_user_data = scaler.transform(new_user_df[selected_features])
    scaled_new_user_df = pd.DataFrame(scaled_new_user_data, columns=selected_features)

    df_combined = pd.concat([scaled_new_user_df, mean_values.loc[lmao].to_frame().T], ignore_index=True)
    print(df_combined)
    max_value = df_combined.values.max() + 0.1
    fig = go.Figure()

    df_combined = df_combined.drop(columns=remove_later)
    for i, row_values in enumerate(df_combined.values):
        if i == 0:
            label = "User Data"
        else:
            label = f"Average User Data for {lmao} Credit Score"
        fig.add_trace(go.Scatterpolar(
            r=list(row_values) + [row_values[0]],  # Repeat the first point after the final point
            theta=list(df_combined.columns) + [df_combined.columns[0]],  # Repeat the first theta after the final theta
            name=label,
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

    return fig
