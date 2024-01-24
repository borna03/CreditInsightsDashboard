import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_correlation_chart(df):
    # Drop unnecessary columns and NaN values
    df_cleaned = df.drop(['ID', 'Customer_ID', 'Month', 'Age'], axis=1).dropna()

    # Encode the target variable (Credit_Score)
    df_cleaned['Credit_Score'] = df_cleaned['Credit_Score'].map({'Poor': 1, 'Standard': 2, 'Good': 3})

    # Encode categorical variables
    encoder = LabelEncoder()
    for column in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[column] = encoder.fit_transform(df_cleaned[column])

    # Calculate correlations with Credit Score, excluding the target itself
    correlation_without_target = df_cleaned.drop('Credit_Score', axis=1).corrwith(
        df_cleaned['Credit_Score']).sort_values()

    # Plotting the diverging bar chart using Plotly
    fig = px.bar(
        x=correlation_without_target,
        y=correlation_without_target.index,  # Use original column names here
        orientation='h',
        color=correlation_without_target.apply(lambda x: 'Positive' if x > 0 else 'Negative'),
        color_discrete_map={'Positive': 'lightblue', 'Negative': 'lightsalmon'}
    )

    fig.update_layout(
        title='Correlations with Credit Scores: Key Influencing Factors',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Variable',
        showlegend=False
    )

    # Customizing hover text with formatted variable names
    hover_labels = correlation_without_target.index.str.replace('_', ' ')
    fig.for_each_trace(lambda t: t.update(hovertemplate='<b>%{y}</b><br>Correlation: %{x}'))

    fig.update_yaxes(tickvals=correlation_without_target.index, ticktext=hover_labels)

    # Display the figure
    return fig


# Categorical variables
def plot_categorical_distribution(dataframe, column_name):
    """
    Function to create a Plotly figure representing the distribution of a specified
    categorical variable from a CSV data file.

    :param dataframe: dataframe to use
    :param column_name: The name of the categorical column to plot.
    :return: A Plotly figure representing the distribution of the specified variable.
    """
    column_name_stripped = column_name.replace('_', ' ')
    if column_name == 'Payment_Behaviour':
        dataframe = dataframe[dataframe[column_name] != "!@9#%8"]

    if column_name in dataframe.columns and dataframe[column_name].dtype == 'object':
        # Counting the occurrences of each unique value in the column
        count_data = dataframe[column_name].value_counts().sort_index()

        fig = go.Figure(go.Bar(
            x=count_data.index,
            y=count_data.values,
            marker=dict(color='lightblue')  # Set bar color as 'lightblue'
        ))
        fig.update_layout(
            title=f'Distribution of {column_name_stripped}',
            xaxis_title=column_name,
            yaxis_title='Count',
            xaxis_type='category'
        )

        # Customizing hover text
        hover_text = f'<b>{column_name_stripped}:</b> %{{x}}<br><b>Number of Individuals:</b> %{{y}}'
        fig.update_traces(hovertemplate=hover_text)

        return fig
    else:
        return "Column not found or not categorical."

# Discrete values
def plot_discrete_distribution(dataframe, column_name):
    """
    Function to plot the distribution of a specific discrete numerical variable in the dataset
    with custom hover text as a line chart, styled to match another chart in the dashboard.

    :param dataframe: Pandas DataFrame containing the dataset.
    :param column_name: The name of the discrete numerical column to plot.
    :param line_color: Color of the line in the plot.
    :return: A Plotly figure representing the distribution of the specified variable.
    """

    column_name_stripped = column_name.replace('_', ' ')
    if column_name == 'Payment_Behaviour':
        dataframe = dataframe[dataframe[column_name] != "!@9#%8"]
    elif column_name == 'Interest_Rate':
        dataframe[column_name] = dataframe[column_name].astype(float).apply(lambda x: int(x) if x.is_integer() else x)

    if column_name in dataframe.columns and dataframe[column_name].dtype in ['int64', 'float64']:
        # Counting the occurrences of each unique value in the column
        count_data = dataframe[column_name].value_counts().sort_index()

        fig = go.Figure(go.Scatter(x=count_data.index, y=count_data.values, mode='lines+markers',
                                   line=dict(color='lightblue'), marker=dict(color='lightblue')))
        fig.update_layout(
            title=f'Distribution of {column_name_stripped}',
            xaxis_title=column_name_stripped,
            yaxis_title='Count'
        )

        # Customizing hover text
        hover_text = f'<b>{column_name_stripped}:</b> %{{x}}<br><b>Number of Individuals:</b> %{{y}}'
        fig.update_traces(hovertemplate=hover_text)

        return fig
    else:
        return "Column not found or not a discrete numerical type."


# Continuous variables

import plotly.graph_objects as go


def plot_continuous_distribution(dataframe, column_name):
    """
    Function to plot the distribution of a specific numerical variable in the dataset
    with custom hover text. Filters out values above 150,000 for Annual_Income.

    :param dataframe: Pandas DataFrame containing the dataset.
    :param column_name: The name of the numerical column to plot.
    :return: A Plotly figure representing the distribution of the specified variable.
    """
    filtered_df = dataframe.copy()
    filtered_df[column_name] = filtered_df[column_name].dropna()

    if column_name in filtered_df.columns and filtered_df[column_name].dtype in ['int64', 'float64']:
        # Apply filter for Annual_Income
        if column_name == 'Annual_Income':
            filtered_df = filtered_df[filtered_df[column_name] <= 150000]

        fig = go.Figure(go.Histogram(x=filtered_df[column_name], marker_color='lightblue'))

        # Customizing hover text
        column_name_stripped = column_name.replace('_', ' ')
        fig.update_traces(hovertemplate='<b>Range:</b> %{x}<br><b>Number of Individuals:</b> %{y}')

        fig.update_layout(
            title=f'Distribution of {column_name_stripped}',
            xaxis_title=column_name_stripped,
            yaxis_title='Count'
        )

        return fig


def plot_loan_distribution(dataframe, bar_color):
    """
    Plot distribution of types of loans.

    :param dataframe: DataFrame containing the data.
    :param bar_color: Color of the bars in the chart.
    :param chart_title: Title of the chart.
    :return: Plotly figure object.
    """
    # Initialize a dictionary to count the frequency of each loan type
    loan_counts = {}

    # Process the 'Type_of_Loan' column to count different loans
    for loan_list in dataframe['Type_of_Loan'].dropna():
        # Replace 'and' with a comma or remove it based on the context
        loan_list = loan_list.replace(' and ', ', ')
        if loan_list not in ['Not specified', 'Unknown']:
            loans = loan_list.split(', ')
            for loan in loans:
                loan_counts[loan] = loan_counts.get(loan, 0) + 1

    # Convert the dictionary to a DataFrame
    loan_counts_df = pd.DataFrame(list(loan_counts.items()), columns=['Loan_Type', 'Count'])

    # Plotting the data
    fig = px.bar(loan_counts_df, x='Loan_Type', y='Count', color_discrete_sequence=[bar_color])

    fig.update_layout(
        title=f'Distribution of Loan Type',
        xaxis_title='Loan Type',
        yaxis_title='Count',
        xaxis_type='category'
    )

    return fig
