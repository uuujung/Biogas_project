# -*- coding: utf-8 -*-
"""
Created on Fri May  1

@author: jip
"""

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import dash_table as dt

import plotly.figure_factory as ff

import pandas as pd

from app import app

import db_io_utils as utils
from datetime import date


raw_data, intp_data, nw_data, nw_intp_data, pred_data, mape_data, variables, v_dict, v_list = utils.make_html_data()

param = utils.get_config()
gap = param['gap']

def generate_data_radio():
    _radio = dcc.RadioItems(
        id = 'data-selector',
        options = [
            {'label': '원자료', 'value': '원자료'},
            {'label': '보간자료', 'value': '보간자료'}],
        value = '원자료',
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
            {'label': '전체', 'value': '전체'},
            {'label': '투입량 vs. 생산량', 'value': '투입량 vs. 생산량'},
            {'label': '투입량 vs. 주요인자', 'value': '투입량 vs. 주요인자'},
            {'label': '주요인자 vs. 생산량', 'value': '주요인자 vs. 생산량'}],
        value = '전체',
        style={
            'display': 'inline-block',
            'margin-left': '0px'
        }
    )
    return _radio

def generate_left_card():
    _left_card = html.Div(
        id = "left-data-card",
        className = 'container',
        children = [
            html.H3("상관분석", style = {"text-align": "left", "font-weight": "bold"}),
            html.Br(),
            html.H5('분석 대상 자료', style = {"text-align": "left", "font-weight": "bold"}),
            generate_data_radio(),
            html.Br(),
            html.Br(),
            html.H5('분석 유형', style = {"text-align": "left", "font-weight": "bold"}),
            generate_type_radio(),
        ],
        style={'text-align': 'left'}        
    )
    return _left_card


def generate_graph_card():
    _right_graph = dcc.Graph(
        id = 'correlation-graph',
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
                    html.H6("요약통계량", style = {"text-align": "center", "font-weight": "bold"}),
                    html.Div(id = 'correlation-table'),
                ],
                style = {"width": "80%"}#, "padding-left": "55px"}#, }
            ),
        ],
    )

    return _card
  

@app.callback(Output("correlation-graph", "figure"), [Input('data-selector', 'value'), Input('type-selector', 'value'),Input('my-date-picker-range', 'start_date')
     ,Input('my-date-picker-range', 'end_date')])
def set_display_figure(selected_table, selected_type, start_date, end_date):

    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d 00:00:00')
   
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d 00:00:00')
    
    
    if selected_table == '원자료':
        df_sub = raw_data
        df_sub = df_sub[start_date_string<= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]
        df_sub = df_sub.iloc[:, 1:].apply(pd.to_numeric)
        
        
    elif selected_table == '보간자료':
        df_sub = intp_data
        df_sub = df_sub[start_date_string<= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]
        df_sub = df_sub.iloc[:, 1:].apply(pd.to_numeric)
        
    corrs = df_sub.corr()
    #print(corrs)
    
    if selected_type == '투입량 vs. 생산량':
        selected_var = [variables['input'][1]] + variables['output']
        selected_var = selected_var
        
        corrs = corrs.loc[selected_var]
    elif selected_type == '투입량 vs. 주요인자':
        selected_var = [variables['input'][1]] + variables['key']
        selected_var = selected_var
        
        corrs = corrs.loc[selected_var]
    elif selected_type == '주요인자 vs. 생산량':
        selected_var = variables['key'] + variables['output']
        selected_var = selected_var
        corrs = corrs.loc[selected_var]
    else:
        pass
        
    figure = ff.create_annotated_heatmap(z = corrs.values,
        x = list(corrs.columns),
        y = list(corrs.index),
        annotation_text = corrs.round(2).values,
        showscale = True)

    return figure
       
  
@app.callback(Output('correlation-table', 'children'),
    [Input('data-selector', 'value'),Input('my-date-picker-range', 'start_date')
     ,Input('my-date-picker-range', 'end_date')])
def set_display_table(selected_table, start_date, end_date ):

    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d 00:00:00')
   
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d 00:00:00')
    
       

    if selected_table == '원자료':
        df_sub = raw_data
        df_sub = df_sub[start_date_string<= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]  
        
    elif selected_table == '보간자료':
        df_sub = intp_data
        df_sub = df_sub[start_date_string<= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]        
    
    
    df_sub_stat_ = df_sub.iloc[:, 1:].astype('float32').describe().transpose()
    tmp_ = pd.DataFrame(df_sub_stat_.index)
    tmp_.rename(columns = {0: '변수'}, inplace = True) 
    tmp_.index = df_sub_stat_.index

    df_sub_stat = pd.concat([tmp_, df_sub_stat_], axis = 1)
    for col in df_sub_stat.columns[2:]:
        df_sub_stat[col] = df_sub_stat[col].map("{:,.2f}".format)
        
    stat_ = dt.DataTable(
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
        page_size = 3,
        columns = [{"name": i, "id": i} for i in df_sub_stat.columns],
        data = df_sub_stat.to_dict("rows"),
    )

    return stat_