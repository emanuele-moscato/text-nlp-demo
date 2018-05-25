from flask import Flask, send_from_directory
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import sys
sys.path.insert(0, '../src')
import pdf_lda
import app_tools
import os
import base64
import pickle
import pdftotext

UPLOAD_DIRECTORY = './app-uploaded-files/'
DATA_DIRECTORY = '../data/'

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

server = Flask('Text NLP demo app', static_url_path='')

# Hack to allow serving custom CSS. Taken from this:
# https://community.plot.ly/t/how-do-i-use-dash-to-add-local-css/4914/4
@server.route('/static/style.css')
def serve_stylesheet():
    return server.send_static_file('style.css')
    
@server.route('/favicon.ico')
def favicon():
    return server.send_static_file('400x400SML-01.png')
    
@server.route('/download/<path:path>')
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
    
app = dash.Dash(
    'Text NLP demo app',
    server=server,
    url_base_pathname='/',
    csrf_protect=False
)
app.config['suppress_callback_exceptions']=True

app.css.append_css({
    'external_url': '/static/style.css'
})

app.title = 'Text NLP demo'

file_selector = dcc.Dropdown(
    id='file-selector',
    options=app_tools.list_for_dropdown(pdf_lda.list_input_files())
)

app.layout = html.Div(
    id='app-container',
    children=[
        html.H1("NLP on text documents"),
        html.H2(html.Li("Topic extraction on documents")),
        html.Div(
            [html.Div(
                [html.Div(
                    id='selector-container',
                    children=[dcc.Dropdown(
                        id='file-selector',
                        options=app_tools.list_for_dropdown(
                            pdf_lda.list_input_files()
                        )
                    )]
                ),
                html.Br(),
                html.P("N-gram range:"),
                dcc.RangeSlider(
                    id='ngram-range-slider',
                    min=1,
                    max=5,
                    step=1,
                    value=[1, 1],
                    marks={
                        1: '1',
                        2: '2',
                        3: '3',
                        4: '4',
                        5: '5'
                    },
                ),
                html.Br(),
                html.Br(),
                html.Div(
                    [html.Button(
                        'Topic analysis',
                        id='compute-button'
                    )],
                    style={'textAlign': 'center'}
                ),
                html.Div(
                    style={'height': '20px'}
                ),
                dcc.Upload(
                    id='upload',
                    children=[
                        'Upload files'
                    ],
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'marginTop': '20px'
                    }
                )],
                style={'marginTop': 12},
                className='four columns'
            ),
            html.Div(
                id='topics-container',
                children=[],
                className='eight columns'
            )],
            className='row'
        ),
        html.Div(style={'height': '40px'}),
        html.H2(
            html.Li("An example of (dimensionally-reduced) word embedding")
        ),
        html.Button("Show a word embedding", id='plot-button'),
        html.Div(
            id='plot-div',
            children = [],
            className = 'row'
        )
    ]
)


@app.callback(
    Output('topics-container', 'children'),
    [Input('compute-button', 'n_clicks')],
    [State('file-selector', 'value'),
        State('ngram-range-slider', 'value')]
)
def compute_topics(n_clicks, filename, ngram_range):
    ngram_range = tuple(ngram_range)
    if not n_clicks:
        return app_tools.generate_markdown([])
    else:
        if filename:
            return app_tools.generate_markdown(
                pdf_lda.extract_topics(filename, ngram_range)
            )
        else:
            return app_tools.generate_markdown([])

@app.callback(
    Output('selector-container', 'children'),
    [Input('upload', 'filename'), Input('upload', 'contents')])
def update_file_list(uploaded_filename, uploaded_file_contents):
    if uploaded_filename is not None and uploaded_file_contents is not None:
        app_tools.save_file(
            uploaded_filename,
            uploaded_file_contents,
            UPLOAD_DIRECTORY
        )
        files_list = pdf_lda.list_input_files()# +[uploaded_filename]
        return dcc.Dropdown(
            id='file-selector',
            options=app_tools.list_for_dropdown(
                files_list
            )
        )
    else:
        return dcc.Dropdown(
            id='file-selector',
            options=app_tools.list_for_dropdown(
                pdf_lda.list_input_files()
            )
        )
        
@app.callback(
    Output('plot-div', 'children'),
    [Input('plot-button', 'n_clicks')]
)
def show_word_embedding_plot(n_clicks):
    if n_clicks is not None:
        with open(DATA_DIRECTORY+'long_contract_2d_vectors.pkl', 'rb') as f:
            reduced_embedded_vectors = pickle.load(f)
        with open(
            os.path.join(DATA_DIRECTORY, 'Exhibit-A-SAMPLE-CONTRACT.pdf'),
            'rb'
        ) as f:
            pdf = pdftotext.PDF(f)
        text = ''.join(pdf)
        trace = go.Scatter(
            x = reduced_embedded_vectors[:,0],
            y = reduced_embedded_vectors[:,1],
            text=text.split(),
            hoverinfo = 'text',
            mode = 'markers'
        )
        layout = go.Layout(
            margin=go.Margin(
                t=25,
                l=20
            )
        )
        fig = go.Figure(data=[trace], layout=layout)
        container = html.Div(
            [dcc.Graph(
                id='word-embedding-plot',
                figure=fig
            )],
            className = 'eight columns'
        )
        return container
    


if __name__ == '__main__':
    server.run(port=8888, debug=True)