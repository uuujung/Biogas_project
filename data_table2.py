# -*- coding: utf-8 -*-
"""
Created on Fri May  1

@author: jip
"""

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import dash_table as dt

import pandas as pd

from app import app

import db_io_utils as utils

from datetime import date
import dash
import re


raw_data, intp_data, nw_data, nw_intp_data, pred_data, mape_data, variables, v_dict, v_list = utils.make_html_data()

def generate_radio():
    _radio = dcc.RadioItems(
        id = 'data-selector',
        options = [
            {'label': '원자료', 'value': '원자료'},
            {'label': '보간자료', 'value': '보간자료'},
            {'label': '예측자료', 'value': '예측자료'}],
        value = '보간자료',
        style={
            'display': 'inline-block',
            'margin-left': '0px'
        }
    )
    return _radio

if __name__ == '__main__':
    app.run_server(debug=True)
def generate_left_card():
    _left_card = html.Div(
        id = "left-card",
        className = 'container',
        children = [
            html.H3("자료 조회", style = {"text-align": "left", "font-weight": "bold"}),
            html.Br(),
            html.H5('유형', style = {"text-align": "left", "font-weight": "bold"}),
            generate_radio(),
            html.Br(),
            html.Br(),
            html.Br(),
        ],
        style={'text-align': 'left'}       
    )
    
   
    return _left_card



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
                    html.Div(
                        
                        children = [
                        
                            html.Label('조회기간 선택 : 시작일    ->    종료일     ', 
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
                        ]
                    ), 
                    html.H6("요약 통계량", style = {"text-align": "center", "font-weight": "bold"}),
                    html.Div(id = 'display-selected-table-stats'),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.H6("측정값", style = {"text-align": "center", "font-weight": "bold"}),
                    html.Div(id = 'display-selected-table'),
                ],
                style = {"width": "80%"}#, "padding-left": "55px"}#, }
            ),
        ],
    )

    return _card
  
       
@app.callback([Output('display-selected-table', 'children'), Output('display-selected-table-stats', 'children')],
    [Input('data-selector', 'value'), Input('my-date-picker-range', 'start_date')
     ,Input('my-date-picker-range', 'end_date')])
def set_display_table(selected_table, start_date, end_date):
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d 00:00:00')
        #start_date_string2 = date.date.strptime(str(start_date_string, '%Y-%m-%d %H:%M:%S'))
        
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d 00:00:00')
        #end_date_string2 = date.date.strptime(str(end_date_string, '%Y-%m-%d %H:%M:%S'))

    if selected_table == '원자료':
        df_sub = raw_data
        df_sub = df_sub[start_date_string <= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]

    elif selected_table == '보간자료':
        df_sub = intp_data
        df_sub = df_sub[start_date_string <= df_sub['Time']]
        df_sub = df_sub[df_sub['Time'] <= end_date_string]
        
    elif selected_table == '예측자료':
        df_sub = pred_data
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


    return data_, stat_


