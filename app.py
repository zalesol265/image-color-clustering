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
                children=html.Div(['Upload image']),
                multiple=False
            ),
            html.Label('Number of Colors:'),
            dcc.Slider(
                id='num-clusters',
                min=2, max=20, step=1, value=5,
                marks={i: str(i) for i in range(2, 21)}
            ),
            html.Label('Pixelation Level:'),
            dcc.Slider(
                id='pixelation-level',
                min=1, max=20, step=1, value=1,
                marks={i: str(i) for i in range(1, 21)}
            ),
        ], width=6),  # Settings in the top-left corner

        dbc.Col([
            html.Div(id='output-image-upload'),
        ], width=6),  # Upload component and input image in the top-right corner
    ], style={'margin-top': '20px'}),

    dbc.Row([
        dbc.Col([
            html.Div(id='color-section'),
        ], width=6),  # Chosen colors on the left

        dbc.Col([
            html.Div(id='cluster-section'),
        ], width=6),  # Clustered image on the right
    ], style={'margin-top': '20px'})  # Row for color list and output image
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


def calculate_colors(image, num_clusters):
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

    # Get unique colors and their counts
    unique_colors, _ = np.unique(clustered, axis=0, return_counts=True)

    return clustered_image_pil, unique_colors


@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def display_uploaded_image(contents):
    if contents:
        uploaded_image = html.Img(src=contents, style={'width': '100%'})
        return uploaded_image
    return None


@app.callback(
    [Output('color-section', 'children'),
     Output('cluster-section', 'children')],
    [Input('num-clusters', 'value'),
     Input('pixelation-level', 'value'),
     Input('upload-image', 'contents')]
)
def update_clustering(num_clusters, pixelation_level, contents):
    if contents:
        # Decode the uploaded image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Pixelate the image
        pixelated_image = pixelate_image(image, pixelation_level)

        # Calculate clustered image and color information
        clustered_image, unique_colors = calculate_colors(pixelated_image, num_clusters)

        # Encode the clustered image to base64
        buffered_clustered = io.BytesIO()
        clustered_image.save(buffered_clustered, format="PNG")
        clustered_base64 = base64.b64encode(buffered_clustered.getvalue()).decode('utf-8')
        clustered_image_html = html.Img(src=f'data:image/png;base64,{clustered_base64}', style={'width': '100%'})

        # Prepare HTML for color list
        color_list_items = []
        for color in unique_colors:
            # Convert color values from float64 to int and then to string
            rgb_values = f"rgb{tuple(map(int, color))}"
            # Create a div with the color square and RGB values
            color_square = html.Div(style={
                'width': '30px', 'height': '30px', 'background-color': f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})'
            })
            color_list_items.append(html.Div([
                color_square,
                html.Span(rgb_values, style={'margin-left': '10px'})
            ], className='color-card'))

        color_section = html.Div([
            html.H3("Chosen Colors"),
            html.Div(color_list_items, style={'display': 'flex', 'flex-wrap': 'wrap'})
        ])

        cluster_section = html.Div([
            html.H3("Clustered Image"),
            clustered_image_html
        ])

        return color_section, cluster_section

    return None, None


if __name__ == '__main__':
    app.run_server(debug=True)
