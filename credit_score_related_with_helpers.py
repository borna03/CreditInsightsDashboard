import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file into a DataFrame
plot_discrete_data = pd.read_csv("full_processed_data_sorted.csv")


# Bar Charts function
def plot_bar_chart(data, category, bar_color,
                   title, x_label, y_label='Average Credit Score'):
    """
    Plots a bar chart of the average credit score against a discrete variable.
    """

    # Map the categorical credit scores to numerical values
    credit_score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
    data['Credit_Score_Num'] = data['Credit_Score'].map(credit_score_mapping)

    # Special handling for Payment_Behaviour and Interest Rate
    if category == 'Payment_Behaviour':
        data = data[data[category] != "!@9#%8"]
    elif category == 'Interest_Rate':
        data[category] = data[category].astype(float).apply(lambda x: int(x) if x.is_integer() else x)

    # Calculate the average credit score for the specified category
    average_credit_score = data.groupby(category)['Credit_Score_Num'].mean().reset_index()

    # Plotting using plotly
    fig = px.bar(average_credit_score, x=category, y='Credit_Score_Num',
                 title=title, labels={category: category, 'Credit_Score_Num': y_label},
                 color_discrete_sequence=[bar_color] * len(average_credit_score))

    fig.add_hline(y=2, line_dash="dash", line_color='rgba(128, 128, 128, 0.5)')

    # Set y-axis range
    fig.update_yaxes(range=[1, 3])

    # Update layout to adjust legend
    fig.update_layout(showlegend=True)

    # Update x-axis and y-axis titles
    # fig.update_xaxes(title=category.replace('_', ' '))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)

    return fig


def plot_regression_line(data, x_column, line_color, y_column, title, x_label):
    """
    Plots a line chart of the average credit score against a discrete variable.
    The line in this chart represents the relationship between credit score and the chosen variable.
    If the x_column is 'Annual_Income', outliers are removed using the IQR method before plotting.
    """
    # Check if x_column is 'Annual_Income' and remove outliers
    if x_column == 'Annual_Income':
        Q1 = data[x_column].quantile(0.25)
        Q3 = data[x_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = data[(data[x_column] >= lower_bound) & (data[x_column] <= upper_bound)]

    elif x_column == 'Total_EMI_per_month':
        filtered_df = data[(data[x_column] >= 0) & (data[x_column] <= 500)]
    else:
        filtered_df = data

    # Create a scatter plot with a trendline (hide the scatter points)
    fig = px.scatter(filtered_df, x=x_column, y=y_column, trendline="ols",
                     labels={x_column: x_column.replace('_', ' '), y_column: 'Average Credit Score'},
                     title=title, range_y=[1, 3])

    # Hide scatter points, only show trendline
    fig.update_traces(marker=dict(opacity=0), selector=dict(mode='markers'))

    fig.add_hline(y=2, line_dash="dash", line_color='rgba(128, 128, 128, 0.5)')

    # Update the layout for the legend
    fig.update_layout(showlegend=True)

    # Customize the hovertemplate to not show the first 3 rows (i.e., OLS trendline info)
    fig.update_traces(hovertemplate='Average Credit Score: %{y}, ' + x_column.replace('_', ' ') + ': %{x}')

    fig.data[1].line.color = line_color

    # Update axis titles
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title='Average Credit Score')

    return fig


def plot_line_chart(data, category, line_color, title, x_label, y_label='Average Credit Score'):
    """
    Plots the average credit score based on different categories using plotly as a line chart.

    Parameters:
    data (DataFrame): The dataset containing the credit score information.
    category (str): The column name in the dataset to group by (e.g., 'Occupation', 'Credit_Mix').
    line_color (str): Color of the line in the plot.
    title (str): The title of the plot.
    y_label (str, optional): Label for the y-axis. Defaults to 'Average Credit Score'.
    horizontal_line (bool, optional): Whether to include a horizontal line representing the average score. Defaults to True.
    :type x_label: string
    """
    # Map the categorical credit scores to numerical values
    credit_score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
    data['Credit_Score_Num'] = data['Credit_Score'].map(credit_score_mapping)

    # Special handling for Payment_Behaviour and Interest Rate
    if category == 'Payment_Behaviour':
        data = data[data[category] != "!@9#%8"]
    elif category == 'Interest_Rate':
        data[category] = data[category].astype(float).apply(lambda x: int(x) if x.is_integer() else x)

    # Calculate the average credit score for the specified category
    average_credit_score = data.groupby(category)['Credit_Score_Num'].mean().reset_index()

    # Plotting using plotly as a line chart
    fig = px.line(average_credit_score, x=category, y='Credit_Score_Num',
                  title=title, labels={category: category, 'Credit_Score_Num': y_label},
                  markers=True, color_discrete_sequence=[line_color])

    # Add horizontal line
    fig.add_hline(y=2, line_dash="dash", line_color='rgba(128, 128, 128, 0.5)')

    # Set y-axis range
    fig.update_yaxes(range=[1, 3])

    # Update layout to adjust legend
    fig.update_layout(showlegend=True)

    # Customize the hovertemplate to not show the first 3 rows (i.e., OLS trendline info)
    # fig.update_traces(hovertemplate='Average Credit Score: %{y}, ' + category.replace('_', ' ') + ': %{x}')
    fig.update_traces(
        hovertemplate='<b>Average Credit Score:</b> %{y}<br><b>' + category.replace('_', ' ') + ':</b> %{x}')

    # Update x-axis and y-axis titles
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)

    return fig


def plot_bar_type_of_loan(dataframe, bar_color, chart_title, y_label='Average Credit Score', x_label='Type of Loan'):
    """
    Plot average credit score by type of loan.

    :param dataframe: DataFrame containing the data.
    :param bar_color: Color of the bars in the chart.
    :param chart_title: Title of the chart.
    :param y_label: Y-axis label.
    :return: Plotly figure object.
    """
    # Map Credit Scores to numeric values
    credit_score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
    dataframe['Credit_Score_Numeric'] = dataframe['Credit_Score'].map(credit_score_mapping)

    # Clean the 'Type_of_Loan' column by removing ' and ' if it appears and also any trailing commas and whitespace
    dataframe['Type_of_Loan'] = (
        dataframe['Type_of_Loan']
        .str.replace(r'\band\s+', '', regex=True)  # remove 'and' followed by any number of spaces
        .str.strip(', ')  # remove leading/trailing commas and whitespace
    )

    # Expand the 'Type_of_Loan' column so each type of loan is a separate row
    expanded_loans = dataframe['Type_of_Loan'].str.split(', ', expand=True).stack().str.strip()
    expanded_loans.name = 'Type_of_Loan'
    dataframe = dataframe.drop('Type_of_Loan', axis=1).join(expanded_loans.reset_index(level=1, drop=True))

    # Remove rows where 'Type_of_Loan' is 'Not Specified' or 'Unknown'
    dataframe = dataframe[~dataframe['Type_of_Loan'].str.lower().isin(['not specified', 'unknown'])]

    # Calculate the average credit score for each loan type
    loan_avg_scores = (
        dataframe.groupby('Type_of_Loan')['Credit_Score_Numeric']
        .mean()
        .reset_index(name='Average_Credit_Score')
    )

    # Plotting the data
    fig = px.bar(
        loan_avg_scores,
        x='Type_of_Loan',
        y='Average_Credit_Score',
        color_discrete_sequence=[bar_color],
        title=chart_title,
        labels={'Average_Credit_Score': y_label, 'Type_of_Loan': x_label}
    )

    fig.add_hline(y=2, line_dash="dash", line_color="grey")
    fig.update_yaxes(range=[0, 3])
    fig.update_xaxes(tickangle=45)

    return fig
