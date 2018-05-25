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

UPLOAD_DIRECTORY = './app-uploaded-files/'

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
        html.H1("Topic extraction on documents"),
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
                html.Button('Topic analysis', id='compute-button'),
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
        html.Div(id='output-div')
    ]
)


@app.callback(
    Output('topics-container', 'children'),
    [Input('compute-button', 'n_clicks')],
    [State('file-selector', 'value')]
)
def compute_topics(n_clicks, filename):
    if not n_clicks:
        return app_tools.generate_markdown([])
    else:
        if filename:
            return app_tools.generate_markdown(pdf_lda.extract_topics(filename))
        else:
            return app_tools.generate_markdown([])

@app.callback(
    Output('selector-container', 'children'),
    [Input('upload', 'filename'), Input('upload', 'contents')])
def update_file_list(uploaded_filename, uploaded_file_contents):
    if uploaded_filename is not None and uploaded_file_contents is not None:
        '''
        data = uploaded_file_contents.encode('utf8').split(b'base64,')[-1]
        with open('./app-uploaded-files/'+uploaded_filename, 'wb') as f:
            f.write(base64.decodebytes(data))
        '''
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
    


if __name__ == '__main__':
    server.run(port=8888, debug=True)