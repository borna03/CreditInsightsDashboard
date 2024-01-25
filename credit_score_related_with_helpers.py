import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file into a DataFrame
plot_discrete_data = pd.read_csv("full_processed_data_sorted.csv")


# Bar Charts function
def plot_bar_chart(data, category, bar_color,
                   title, y_label='Average Credit Score',):
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
    fig.update_xaxes(title=category.replace('_', ' '))
    fig.update_yaxes(title=y_label)

    return fig


def plot_regression_line(data, x_column, line_color, y_column, title):
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
        data = data[(data[x_column] >= lower_bound) & (data[x_column] <= upper_bound)]

    # Create a scatter plot with a trendline (hide the scatter points)
    fig = px.scatter(data, x=x_column, y=y_column, trendline="ols",
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
    fig.update_xaxes(title=x_column.replace('_', ' '))
    fig.update_yaxes(title='Average Credit Score')

    return fig


def plot_line_chart(data, category, line_color, title, y_label='Average Credit Score'):
    """
    Plots the average credit score based on different categories using plotly as a line chart.

    Parameters:
    data (DataFrame): The dataset containing the credit score information.
    category (str): The column name in the dataset to group by (e.g., 'Occupation', 'Credit_Mix').
    line_color (str): Color of the line in the plot.
    title (str): The title of the plot.
    y_label (str, optional): Label for the y-axis. Defaults to 'Average Credit Score'.
    horizontal_line (bool, optional): Whether to include a horizontal line representing the average score. Defaults to True.
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
    fig.update_traces(hovertemplate='<b>Average Credit Score:</b> %{y}<br><b>' + category.replace('_', ' ') + ':</b> %{x}')


    # Update x-axis and y-axis titles
    fig.update_xaxes(title=category.replace('_', ' '))
    fig.update_yaxes(title=y_label)

    return fig


def plot_bar_type_of_loan(dataframe, bar_color, chart_title, y_label='Average Credit Score'):
    """
    Plot average credit score by type of loan.

    :param dataframe: DataFrame containing the data.
    :param bar_color: Color of the bars in the chart.
    :param chart_title: Title of the chart.
    :return: Plotly figure object.
    """
    # Map Credit Scores to numeric values
    credit_score_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
    dataframe['Credit_Score_Numeric'] = dataframe['Credit_Score'].map(credit_score_mapping)

    # Process the 'Type_of_Loan' column to separate different loans
    loan_types = set()
    for loan_list in dataframe['Type_of_Loan'].dropna():
        # Replace 'and' with ','
        loan_list = loan_list.replace(' and ', ', ')
        loans = loan_list.split(', ')
        for loan in loans:
            loan_types.add(loan)

    # Create a dictionary to hold average credit scores for each loan type
    loan_avg_scores = {}
    for loan_type in loan_types:
        # Filter dataframe for each loan type
        # Ensure we replace 'and' in the filtering process as well
        filtered_df = dataframe[dataframe['Type_of_Loan'].str.contains(loan_type.replace(' and ', ', '), na=False)]
        avg_score = filtered_df['Credit_Score_Numeric'].mean()
        loan_avg_scores[loan_type] = avg_score

    # Convert the dictionary to a DataFrame and remove 'and' from the 'Loan_Type'
    loan_avg_scores_df = pd.DataFrame(list(loan_avg_scores.items()), columns=['Loan_Type', 'Average_Credit_Score'])
    loan_avg_scores_df['Loan_Type'] = loan_avg_scores_df['Loan_Type'].str.replace(' and ', ', ')

    # Plotting the data
    fig = px.bar(loan_avg_scores_df, x='Loan_Type', y='Average_Credit_Score', color_discrete_sequence=[bar_color],
                 title=chart_title, labels={'Average_Credit_Score': y_label})

    fig.add_hline(y=2, line_dash="dash", line_color='rgba(128, 128, 128, 0.5)')

    # Set y-axis range and update x-axis tick labels
    fig.update_yaxes(range=[1, 3])
    fig.update_xaxes(tickangle=-45)

    return fig
