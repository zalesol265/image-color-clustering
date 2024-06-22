import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
from PIL import Image, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    html.H1("Image Color Clustering", className="text-center"),  # Title centered
    dbc.Row([
        dbc.Col([
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
                min=2, max=25, step=1, value=5,
                marks={i: str(i) for i in range(2, 26)}
            ),
            html.Label('Pixelation Level:'),
            dcc.Slider(
                id='pixelation-level',
                min=1, max=25, step=1, value=1,
                marks={i: str(i) for i in range(1, 26)}
            ),
        ], width=6, align='center'),  # Upload and controls centered in a 6-column width
    ], justify='center', style={'margin-top': '20px'}),  # Row centered with margin top

    dbc.Row([
        dbc.Col([
            html.H3("Original Image"),
            html.Div(id='output-image-upload'),
        ], width=6, align='center'),  # Original Image centered in a 6-column width
        dbc.Col([
            html.H3("Clustered Image"),
            html.Div(id='output-image-cluster'),
        ], width=6, align='center'),  # Clustered Image centered in a 6-column width
    ], justify='center', style={'margin-top': '20px'})  # Row centered with margin top
])

def pixelate_image(image, pixel_size):
    width, height = image.size
    # Calculate number of pixels in each direction to get average color
    x_blocks = width // pixel_size
    y_blocks = height // pixel_size
    # Resize image to the number of pixels
    pixelated = image.resize((x_blocks, y_blocks), Image.NEAREST)
    # Scale up the image to the original size
    pixelated = pixelated.resize((width, height), Image.NEAREST)
    return pixelated

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    if contents is not None:
        # Decode the uploaded image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        return html.Img(src=contents, style={'width': '100%'})

@app.callback(
    Output('output-image-cluster', 'children'),
    [Input('upload-image', 'contents'), Input('num-clusters', 'value'), Input('pixelation-level', 'value')]
)
def update_clustered_image(contents, num_clusters, pixelation_level):
    if contents is not None:
        # Decode the uploaded image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Pixelate the image
        pixelated_image = pixelate_image(image, pixelation_level)

        # Convert pixelated image to numpy array
        pixelated_array = np.array(pixelated_image)
        h, w, _ = pixelated_array.shape
        pixelated_array = pixelated_array.reshape((h * w, 3))

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(pixelated_array)
        clustered = kmeans.cluster_centers_[kmeans.labels_]

        # Convert back to image format
        clustered_img = clustered.reshape((h, w, 3)).astype('uint8')
        clustered_image_pil = Image.fromarray(clustered_img)

        # Encode the clustered image to base64
        buffered_clustered = io.BytesIO()
        clustered_image_pil.save(buffered_clustered, format="PNG")
        clustered_base64 = base64.b64encode(buffered_clustered.getvalue()).decode('utf-8')

        return html.Img(src=f'data:image/png;base64,{clustered_base64}', style={'width': '100%'})

if __name__ == '__main__':
    app.run_server(debug=True)
