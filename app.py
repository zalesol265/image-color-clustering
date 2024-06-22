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
    html.H1("Image Color Clustering", className="text-center"),
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
    ], justify='center', style={'margin-top': '20px'}),  # Row centered with margin top

    dbc.Row([
        dbc.Col([
            html.Div(id='color-list')
        ], width=12, align='center')  # Color list centered in a 12-column width
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

def calculate_colors(image, num_clusters):
    # Convert image to numpy array
    img_array = np.array(image)
    h, w, _ = img_array.shape
    img_array = img_array.reshape((h * w, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(img_array)
    clustered_centers = kmeans.cluster_centers_

    # Convert back to image format
    clustered_img = clustered_centers[kmeans.labels_].reshape((h, w, 3)).astype('uint8')
    clustered_image_pil = Image.fromarray(clustered_img)

    # Get unique colors and their counts
    unique_colors, color_counts = np.unique(clustered_centers, axis=0, return_counts=True)

    return clustered_image_pil, unique_colors, color_counts

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
    [Output('output-image-cluster', 'children'), Output('color-list', 'children')],
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

        # Calculate clustered image and color information
        clustered_image, unique_colors, color_counts = calculate_colors(pixelated_image, num_clusters)

        # Encode the clustered image to base64
        buffered_clustered = io.BytesIO()
        clustered_image.save(buffered_clustered, format="PNG")
        clustered_base64 = base64.b64encode(buffered_clustered.getvalue()).decode('utf-8')

        # Prepare HTML for clustered image
        clustered_image_html = html.Img(src=f'data:image/png;base64,{clustered_base64}', style={'width': '100%'})

        # Prepare HTML for color list
        color_list_items = []
        for color, count in zip(unique_colors, color_counts):
            # Convert color values from float64 to int and then to string
            rgb_values = f"RGB: {tuple(map(int, color))}"
            # Create a div with the color square and RGB values
            color_square = html.Div(className='color-box', style={'width': '30px', 'height': '30px', 'background-color': f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})'})
            color_list_items.append(html.Div([color_square, rgb_values], style={'display': 'inline-block', 'margin': '10px'}))

        color_list_html = html.Div(color_list_items)

        return clustered_image_html, color_list_html

    return None, None


if __name__ == '__main__':
    app.run_server(debug=True)
