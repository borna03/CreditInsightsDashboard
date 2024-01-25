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
        color_discrete_map={'Positive': 'lightblue', 'Negative': 'palevioletred'}
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
    Function to create a Plotly figure representing the density distribution of a specified
    categorical variable from a CSV data file, differentiated by credit score.

    :param dataframe: dataframe to use
    :param column_name: The name of the categorical column to plot.
    :return: A Plotly figure representing the density distribution of the specified variable.
    """
    # Define the color mapping
    color_map = {'Poor': 'palevioletred', 'Standard': 'grey', 'Good': 'lightblue'}

    column_name_stripped = column_name.replace('_', ' ')

    if column_name in dataframe.columns and dataframe[column_name].dtype == 'object':
        # Create a grouped dataframe by `column_name` and `Credit_Score`
        grouped_data = dataframe.groupby([column_name, 'Credit_Score']).size().unstack(fill_value=0)

        # Calculate density instead of count for each credit score
        densities = grouped_data.div(grouped_data.sum(axis=0), axis=1)

        # Create a bar for each credit score category
        bars = []
        for credit_score in color_map.keys():
            bars.append(go.Bar(
                name=credit_score,
                x=densities.index,
                y=densities[credit_score],
                marker=dict(color=color_map[credit_score]),  # Use the color map for bar colors
            ))

        # Customizing hover text to show the specific category and density in percentage
        for bar in bars:
            bar.hovertemplate = f'<b>{column_name_stripped}:</b> %{{x}}<br><b>Density:</b> %{{y:.2%}}'

        # Use `go.Figure` to create the stacked bar chart
        fig = go.Figure(data=bars)
        fig.update_layout(
            barmode='stack',
            title=f'Density Distribution of {column_name_stripped} by Credit Score',
            xaxis_title=column_name_stripped,
            yaxis_title='Density',
            legend_title='<b>Credit Score</b>',
            yaxis=dict(tickformat=',.0%')  # Format y-axis as a percentage
        )

        # Customizing hover text to show percentage


        return fig
    else:
        return "Column not found or not categorical."

def plot_discrete_distribution(dataframe, column_name):
    """
    Function to plot the density distribution of a specific discrete numerical variable in the dataset
    with custom hover text as a line chart, styled to match another chart in the dashboard.

    :param dataframe: Pandas DataFrame containing the dataset.
    :param column_name: The name of the discrete numerical column to plot.
    :return: A Plotly figure representing the density distribution of the specified variable.
    """
    # Color map for different credit scores
    color_map = {'Poor': 'palevioletred', 'Standard': 'grey', 'Good': 'lightblue'}

    # Set up the figure
    fig = go.Figure()

    # Plot each credit score category as a separate line
    for score, color in color_map.items():
        # Filter the dataframe for the current category
        category_df = dataframe[dataframe['Credit_Score'] == score]

        column_name_stripped = column_name.replace('_', ' ')
        # Check if the dataframe is not empty
        if not category_df.empty:
            # Count occurrences of each unique value in the column
            count_data = category_df[column_name].value_counts().sort_index()

            # Calculate the density for each value
            density_data = count_data / count_data.sum()

            # Add a trace for this category
            fig.add_trace(go.Scatter(
                x=density_data.index,
                y=density_data.values,
                mode='lines+markers',
                line=dict(color=color),
                marker=dict(color=color),
                name=score  # Use the credit score category as the trace name
            ))

    # Update layout
    fig.update_layout(
        title=f'Density Distribution of {column_name_stripped} by Credit Score',
        xaxis_title=column_name_stripped,
        yaxis_title='Density',
        legend_title='Credit Score',
        yaxis=dict(tickformat='.2%')  # Format y-axis as a percentage
    )

    # Customizing hover text to show density
    hover_text = f'<b>{column_name_stripped}:</b> %{{x}}<br><b>Density:</b> %{{y:.2%}}'
    fig.update_traces(hovertemplate=hover_text)

    return fig


def plot_continuous_distribution(dataframe, column_name):
    """
    Function to plot a normalized density distribution of a specific numerical variable in the dataset
    with custom hover text, for all data or separate by credit score categories.

    :param dataframe: Pandas DataFrame containing the dataset.
    :param column_name: The name of the numerical column to plot.
    :return: A Plotly figure representing the normalized density distribution of the specified variable.
    """
    filtered_df = dataframe.copy()
    filtered_df[column_name] = filtered_df[column_name].dropna()
    # Apply filter for Annual_Income
    if column_name == 'Annual_Income':
        filtered_df = filtered_df[filtered_df[column_name] <= 150000]
    elif column_name == 'Total_EMI_per_month':
        filtered_df = filtered_df[(filtered_df[column_name] >= 0) & (filtered_df[column_name] <= 500 )]

    color_map = {'Poor': 'palevioletred', 'Standard': 'grey', 'Good': 'lightblue'}

    fig = go.Figure()

    # Plot each category separately with normalization
    for score, color in color_map.items():
        category_df = dataframe[dataframe['Credit_Score'] == score]  # filter_by_credit_score function assumed
        fig.add_trace(go.Histogram(
            x=category_df[column_name],
            marker_color=color,
            xbins=dict(start=np.floor(category_df[column_name].min()),
                       end=np.ceil(category_df[column_name].max())),
            name=score,
            histnorm='probability density'  # Normalize the histogram
        ))

    # Specify the layout properties
    fig.update_layout(
        title=f'Density Distribution of {column_name.replace("_", " ")}',
        xaxis_title=column_name.replace("_", " "),
        yaxis_title='Density',  # Updated y-axis title to 'Density'
        xaxis=dict(showline=True, showgrid=True, zeroline=False),
        yaxis=dict(showline=True, showgrid=True, zeroline=False),
        barmode='overlay',  # Overlap the histograms
        legend_title='Credit Score'
    )
    column_name_stripped = column_name.replace('_', ' ')
    fig.update_traces(hovertemplate=f'<b>{column_name_stripped}:</b> %{{x}}<br><b>Density:</b> %{{y:.2%}}')


    # Reduce opacity to see overlapping bars
    fig.update_traces(opacity=0.6)

    return fig


import plotly.graph_objects as go
import pandas as pd

def plot_loan_distribution(dataframe):
    """
    Plot distribution of types of loans for all credit scores.

    :param dataframe: DataFrame containing the data.
    :return: Plotly figure object.
    """
    # Define the color mapping
    color_map = {'Poor': 'palevioletred', 'Standard': 'grey', 'Good': 'lightblue'}

    # Initialize a dictionary to count the frequency of each loan type by credit score
    loan_counts = {cs: {} for cs in color_map.keys()}

    # Process the 'Type_of_Loan' column to count different loans by credit score
    for _, row in dataframe.dropna(subset=['Type_of_Loan', 'Credit_Score']).iterrows():
        loans = row['Type_of_Loan'].replace(' and ', ', ').split(', ')
        credit_score = row['Credit_Score']
        for loan in loans:
            if loan not in ['Not specified', 'Unknown']:
                loan_counts[credit_score][loan] = loan_counts[credit_score].get(loan, 0) + 1

    # Convert the nested dictionary to a DataFrame for plotting
    loan_counts_df = pd.DataFrame(loan_counts).fillna(0).reset_index().rename(columns={'index': 'Loan_Type'})

    # Create a bar for each credit score category
    bars = []
    for credit_score, color in color_map.items():
        bars.append(go.Bar(
            name=credit_score,
            x=loan_counts_df['Loan_Type'],
            y=loan_counts_df[credit_score],
            marker=dict(color=color),
        ))

    # Use `go.Figure` to create the stacked bar chart
    fig = go.Figure(data=bars)
    fig.update_layout(
        barmode='stack',
        title='Distribution of Loan Type Across All Credit Scores',
        xaxis_title='Loan Type',
        yaxis_title='Count',
        legend_title='<b>Credit Score</b>',
        xaxis={'categoryorder': 'total descending'}  # This will order the categories by total count
    )

    # Customizing hover text
    hover_text = '<b>Loan Type:</b> %{x}<br><b>Number of Individuals:</b> %{y}<br><b>Credit Score:</b> %{data.name}'
    fig.update_traces(hovertemplate=hover_text)

    return fig

def filter_by_credit_score(dataframe, filter_credit_score):
    if filter_credit_score == 'poor':
        dataframe = dataframe[dataframe['Credit_Score'] == 'Poor']
    elif filter_credit_score == 'standard':
        dataframe = dataframe[dataframe['Credit_Score'] == 'Standard']
    elif filter_credit_score == 'excellent':
        dataframe = dataframe[dataframe['Credit_Score'] == 'Good']
    elif filter_credit_score == 'default':
        dataframe = dataframe
    return dataframe
