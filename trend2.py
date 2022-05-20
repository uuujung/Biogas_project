# -*- coding: utf-8 -*-
"""
Created on Fri May  1

@author: jip
"""

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import dash_table as dt

import plotly.graph_objs as go

from app import app

import db_io_utils as utils

from datetime import date
import dash
import re



raw_data, intp_data, nw_data, nw_intp_data, pred_data, mape_data, variables, v_dict, v_list = utils.make_html_data()

def get_options(variables):
    dict_list = []
    for key in variables.keys():
        for var in variables[key]:
            if var != 'Time':
                dict_list.append({'label': var, 'value': var})

    #dict_list.append({'label': 'Disabled', 'value': 'Disabled'})
    return dict_list


def generate_variable_radio():
    _radio = dcc.RadioItems(
        id = 'variable-selector',
        options = get_options(variables),
        value = 'OLR(kg VS/m3)',
        style={
            'display': 'inline-block',
            'margin-left': '0px'
        }
    )
    return _radio


def generate_type_radio():
    _radio = dcc.RadioItems(
        id = 'type-selector',
        options = [
            {'label': 'Nadaraya–Watson', 'value': 'Nadaraya–Watson'}],
        value = 'Nadaraya–Watson',
        style={
            'display': 'inline-block',
            'margin-left': '0px'
        }
    )
    return _radio


def generate_left_card():
    _left_card = html.Div(
        id = "left-card",
        className = 'row',
        children = [
            html.H3("추세분석", style = {"text-align": "left", "font-weight": "bold"}),
            html.Br(),
            html.H5('분석 대상 변수', style = {"text-align": "left", "font-weight": "bold"}),
            generate_variable_radio(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H5('추세 예측 방법', style = {"text-align": "left", "font-weight": "bold"}),
            generate_type_radio()
            
        ],
        style={'text-align': 'left'}        
    )
    return _left_card




def generate_graph_card():
    _right_graph = dcc.Graph(
        id = 'trend-graph',
        config = {'staticPlot': False, 'displayModeBar': False}
    )
    return _right_graph



def generate_table():
    tbl = dt.DataTable(id = 'table')
    return tbl
    

def card():

    _card = html.Div(
        id = "pred-container",
        children = [
            # Left column
            html.Div(
                id = "left-column",
                className = "three columns",
                children = generate_left_card(),
                style = {"width": "17%", "padding-left": "0px", "padding-top": "25px"}
            ),
            # Right column
            html.Div(
                id = "right-column",
                className = "nine columns",
                children = [
                        html.Label('분석기간 선택 : 시작일    ->    종료일     ', 
                                   htmlFor = 'my-date-picker-range', 
                                   style = {"text-align": "right", "font-weight": "bold", 'font-size': '19px', 'color' : 'darkgray'}),
                        dcc.DatePickerRange(
                            id='my-date-picker-range',
                            style = {"width": "100%", "padding-left": "0px", "padding-top": "0px", 'textAlign': 'right'},
                            #style={'marginTop': 10, 'marginBottom': 15},
                            min_date_allowed=date(2019, 1, 1),
                            max_date_allowed=date(2030, 9, 19),
                            initial_visible_month=date(2019, 1, 1),
                            start_date = date(2019, 1, 1),
                            end_date = date(2019, 9, 30)
                        ),
                        html.Div(id='output-container-date-picker-range'),
                        
                    generate_graph_card(),
                    html.H6("자료 상세", style = {"text-align": "center", "font-weight": "bold"}),
                    html.Div(id = 'data-table'),
                ],
                style = {"width": "80%"}#, "padding-left": "55px"}#, }
            ),
        ],
    )

    return _card
  

@app.callback(Output("trend-graph", "figure"), [Input('variable-selector', 'value'), Input('type-selector', 'value'), Input('my-date-picker-range', 'start_date')
     ,Input('my-date-picker-range', 'end_date')])
def set_display_figure(selected_var, selected_type, start_date, end_date):
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d 00:00:00')
   
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d 00:00:00')

    if selected_type == 'Nadaraya–Watson':
        df_sub = nw_data
        df_sub = df_sub[start_date_string <= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]


    df_sub_raw = raw_data
    df_sub_raw = df_sub_raw[start_date_string <= df_sub_raw['Time']]
    df_sub_raw = df_sub_raw[df_sub_raw['Time'] <= end_date_string]    

    figure = go.Figure()
    
    # Add traces
    figure.add_trace(go.Scatter(x = df_sub_raw['Time'], y = df_sub_raw[selected_var],
                        mode='lines+markers',
                        name='측정값'))
    figure.add_trace(go.Scatter(x = df_sub['Time'], y = df_sub[selected_var],
                        mode='lines',
                        name='예측추세'))

    return figure
       
  
@app.callback(Output('data-table', 'children'),
    [Input('type-selector', 'value'), Input('my-date-picker-range', 'start_date')
     ,Input('my-date-picker-range', 'end_date')])
def set_display_table(selected_type, start_date, end_date):
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d 00:00:00')
   
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d 00:00:00')    

    if selected_type == 'Nadaraya–Watson':
        df_sub = nw_data
        df_sub = df_sub[start_date_string <= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]

    
    data_ = dt.DataTable(
        id = "table",
        sort_action = "native",
        row_deletable = False,
        style_data = {"whiteSpace": "normal"},
        style_cell = {
            "padding": "15px",
            "midWidth": "0px",
            "width": "25%",
            "textAlign": "center",
            "border": "white",
        },
        page_current = 0,
        page_size = 10,
        columns = [{"name": i, "id": i} for i in df_sub.columns],
        data = df_sub.to_dict("rows"),
    )

    return data_

