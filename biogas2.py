import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import pathlib

import prediction2 as prediction
import trend2 as trend
import correlation2 as correlation
import data_table2 as data_table 

from app import app

import logging
import win32gui, win32con


#hide = win32gui.GetForegroundWindow()
#win32gui.ShowWindow(hide, win32con.SW_HIDE)

#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)


server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


tabs_styles = {
    'height': '35px',
    'width': '600px'
    
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    # Banner
    html.Div(
        id="banner",
        className="banner",
        children=[html.Img(src = app.get_asset_url("ternary_lab_logo.jpg"))],
    ),
    dcc.Tabs(
        id = "tabs-styled-with-inline", 
        value = 'pred-tab', 
        children = [
            dcc.Tab(label='예측', value = 'pred-tab', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label='추세', value = 'trend-tab', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label='상관분석', value ='corr-tab', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label='자료조회', value ='data-tab', style = tab_style, selected_style = tab_selected_style),
        ], 
        style=tabs_styles
    ),
    html.Div(id = 'tabs-content-inline')
])


@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'pred-tab':
        return prediction.card()
    elif tab == 'trend-tab':
        return trend.card()
    elif tab == 'corr-tab':
        return correlation.card()
    elif tab == 'data-tab':
        return data_table.card()

                                  
if __name__ == '__main__':
    
    app.run_server(debug = True)