import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Image Color Clustering"),
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Label('Number of Colors:'),
            dcc.Slider(
                id='num-clusters',
                min=2, max=20, step=1, value=5,
                marks={i: str(i) for i in range(2, 21)}
            ),
            dbc.Button("Cluster Image", id="cluster-button", color="primary", style={"margin-top": "10px"})
        ], width=4),
        dbc.Col([
            html.H3("Original Image"),
            html.Img(id='original-image', style={'width': '100%'}),
            html.H3("Clustered Image"),
            html.Img(id='clustered-image', style={'width': '100%'})
        ], width=8)
    ])
])

def parse_contents(contents, num_clusters):
    # Decode the uploaded image
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Convert image to numpy array
    img_array = np.array(image)
    h, w, _ = img_array.shape
    img_array = img_array.reshape((h * w, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(img_array)
    clustered = kmeans.cluster_centers_[kmeans.labels_]

    # Convert back to image format
    clustered_img = clustered.reshape((h, w, 3)).astype('uint8')
    clustered_image_pil = Image.fromarray(clustered_img)

    # Encode the original and clustered images to base64
    buffered_original = io.BytesIO()
    image.save(buffered_original, format="PNG")
    original_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

    buffered_clustered = io.BytesIO()
    clustered_image_pil.save(buffered_clustered, format="PNG")
    clustered_base64 = base64.b64encode(buffered_clustered.getvalue()).decode('utf-8')

    return original_base64, clustered_base64

@app.callback(
    [Output('original-image', 'src'), Output('clustered-image', 'src')],
    [Input('cluster-button', 'n_clicks')],
    [State('upload-image', 'contents'), State('num-clusters', 'value')]
)
def update_output(n_clicks, contents, num_clusters):
    if contents is not None:
        original_base64, clustered_base64 = parse_contents(contents, num_clusters)
        original_src = f'data:image/png;base64,{original_base64}'
        clustered_src = f'data:image/png;base64,{clustered_base64}'
        return original_src, clustered_src
    return None, None

if __name__ == '__main__':
    app.run_server(debug=True)
