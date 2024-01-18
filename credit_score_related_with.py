import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file into a DataFrame
plot_discrete_data = pd.read_csv("full_processed_data_sorted.csv")


# Bar Charts function
def plot_bar_chart(data, category, bar_color,
                             title, y_label='Average Credit Score',
                             horizontal_line=True):
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

    # Add horizontal line if needed
    if horizontal_line:
        fig.add_hline(y=2, line_dash="dash", line_color='rgba(0, 0, 255, 0.3)')

    # Set y-axis range
    fig.update_yaxes(range=[1, 3])

    # Update layout to adjust legend
    fig.update_layout(showlegend=True)

    # Update x-axis and y-axis titles
    fig.update_xaxes(title=category.replace('_', ' '))
    fig.update_yaxes(title=y_label)

    return fig


def plot_regression_line(data, x_column, y_column, line_color, title):
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

    fig.add_hline(y=2, line_dash="dash", line_color='rgba(0, 0, 255, 0.5)')

    # Update the layout for the legend
    fig.update_layout(showlegend=True)

    # Customize the hovertemplate to not show the first 3 rows (i.e., OLS trendline info)
    fig.update_traces(hovertemplate='Average Credit Score: %{y}, ' + x_column.replace('_', ' ') + ': %{x}')

    fig.data[1].line.color = line_color

    # Update axis titles
    fig.update_xaxes(title=x_column.replace('_', ' '))
    fig.update_yaxes(title='Average Credit Score')

    return fig


def plot_line_chart(data, category, line_color, title, y_label='Average Credit Score',
                                  horizontal_line=True):
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

    # Add horizontal line if needed
    if horizontal_line:
        fig.add_hline(y=2, line_dash="dash", line_color='rgba(0, 0, 255, 0.5)')

    # Set y-axis range
    fig.update_yaxes(range=[1, 3])

    # Update layout to adjust legend
    fig.update_layout(showlegend=True)

    # Update x-axis and y-axis titles
    fig.update_xaxes(title=category.replace('_', ' '))
    fig.update_yaxes(title=y_label)

    return fig

