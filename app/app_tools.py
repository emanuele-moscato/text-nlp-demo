import dash_core_components as dcc
from textwrap import dedent
import os
import base64

UPLOAD_DIR = './app-uploaded-files/'

def list_for_dropdown(file_list):
    list_for_dropdown = []
    for file_name in file_list:
        list_for_dropdown.append({'label': file_name, 'value': file_name})
    return list_for_dropdown

def generate_markdown(topics_list):
    markdown_string = "#### Topics"
    if len(topics_list)>0:
        for i in range(len(topics_list)):
            new_line = f'\n{i+1}. '+topics_list[i]
            markdown_string = markdown_string+new_line
    markdown = dcc.Markdown(
        dedent(markdown_string)
    )
    return markdown
    
def save_file(name, content, upload_dir):
    data = content.encode('utf8').split(b'base64,')[-1]
    with open(os.path.join(upload_dir, name), 'wb') as fp:
        fp.write(base64.decodebytes(data))