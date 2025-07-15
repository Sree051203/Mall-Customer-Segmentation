import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

df = pd.read_csv("Mall_Customers_Clustered.csv")

app = dash.Dash(__name__)

fig = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Cluster',
    hover_data=['Gender', 'Age'],
    title='Customer Segmentation'
)

app.layout = html.Div([
    html.H1("Mall Customer Segmentation Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)

