from django.shortcuts import render
import traceback
from typing import Optional
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib
import base64
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


sqlite_db_file_path = "./../tvm_pipeline_analytics/db.sqlite3"



def home(request):
    # return HttpResponse("<h1>Hello World</h1>")
    return render(request, 'index.html')

def simple_analytics(request):
    return render(request, 'simple_analytics.html')

def get_simple_analytics_data(request):
    if request.method == 'POST':
        input_start_date = request.POST['start_date'].strip()
        input_end_date = request.POST['end_date'].strip()
        print("input_start_date:", input_start_date)
        print("input_end_date:", input_end_date)

    sqlite_conn = sqlite3.connect(sqlite_db_file_path)
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("select * from proc_daily_tasks where date(created_at) between date('" + input_start_date + "') and date('" + input_end_date + "')")
    df_proc_daily_tasks = pd.DataFrame(sqlite_cursor.fetchall())
    cols = [column[0] for column in sqlite_cursor.description]
    df_proc_daily_tasks.columns = cols
    print("df_proc_daily_tasks.shape:", df_proc_daily_tasks.shape)
    df_proc_daily_tasks.sort_values(by=['created_at'], inplace=True, ascending=True)

    # Delete data if any row contains data of start task of previous day
    if "completed" in df_proc_daily_tasks['message'].iloc[0]:
        df_proc_daily_tasks = df_proc_daily_tasks[1:]
    if "started" in df_proc_daily_tasks['message'].iloc[-1]:
        df_proc_daily_tasks = df_proc_daily_tasks[:-1]
    # print(df_proc_daily_tasks.head(50))

    df_proc_daily_tasks = df_proc_daily_tasks.assign(idx=df_proc_daily_tasks.groupby('message').cumcount()).pivot(index='idx', columns='message', values='created_at').reset_index()
    df_proc_daily_tasks = df_proc_daily_tasks[['proc_daily_tasks started', 'proc_daily_tasks completed']]
    cols = ['proc_daily_tasks started', 'proc_daily_tasks completed']
    df_proc_daily_tasks[cols] = df_proc_daily_tasks[cols].apply(pd.to_datetime)
    df_proc_daily_tasks['pipeline_time_taken'] = pd.to_datetime(df_proc_daily_tasks['proc_daily_tasks completed'].sub(df_proc_daily_tasks['proc_daily_tasks started']).dt.total_seconds(), unit='s').dt.time
    df_proc_daily_tasks = df_proc_daily_tasks.sort_values(by=['proc_daily_tasks started'], kind='mergesort', ascending=False)
    print(df_proc_daily_tasks.head(5))

    temp_proc_daily_tasks = df_proc_daily_tasks[['proc_daily_tasks started', 'pipeline_time_taken']].copy()
    temp_proc_daily_tasks['proc_daily_tasks started'] = pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']).dt.date
    temp_proc_daily_tasks['pipeline_time_taken'] = pd.to_timedelta(pd.to_datetime(temp_proc_daily_tasks['pipeline_time_taken'].astype('string')).dt.time.astype('string'))
    temp_proc_daily_tasks['pipeline_time_taken_seconds'] = temp_proc_daily_tasks['pipeline_time_taken'].dt.total_seconds()
    temp_proc_daily_tasks = temp_proc_daily_tasks.groupby('proc_daily_tasks started', as_index=False).agg({'pipeline_time_taken_seconds': 'mean'})
    temp_proc_daily_tasks = temp_proc_daily_tasks.sort_values(by=['proc_daily_tasks started'], kind='mergesort', ascending=False)
    temp_proc_daily_tasks['pipeline_time_taken_hour'] = temp_proc_daily_tasks['pipeline_time_taken_seconds'] / 3600
    print(temp_proc_daily_tasks.head(5))

    fig, ax = plt.subplots(figsize=(22, 8))
    x = list(temp_proc_daily_tasks['proc_daily_tasks started'])
    y = list(temp_proc_daily_tasks['pipeline_time_taken_hour'])
    y = y[::-1]
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.scatter(range(len(x)), y)
    coef = np.polyfit(range(len(x)), y, 1)
    poly1d_fn = np.poly1d(coef)
    regression_points = poly1d_fn(range(len(x)))
    val_1 = regression_points[0]
    val_n = regression_points[-1]
    percentage_change = ((val_n - val_1) / val_1) * 100
    # display(HTML('<b>Percentage change:  <span style="color:red">' + str(round(percentage_change, 2)) + '</span></b>'))
    ax.plot(range(len(x)), y, regression_points, '--', label="Proc_Daily_Task Runtime")
    ax.axis('tight')
    plt.title('Proc_Daily_Task Runtime between dates ' + r"$\bf{" + input_start_date + "}$" + ' and ' + r"$\bf{" + input_end_date + "}$")
    plt.xlabel('Dates')
    plt.ylabel('Hours')
    plt.legend(loc='upper right')
    # Create names on the x axis
    x_ticks = sorted(list(set(list(pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']).dt.date))))
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    fig.tight_layout()
    # plt.show()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_proc_daily_task_runtime = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()




    sqlite_cursor.execute("select * from messages_log where lower(severity) = 'critical' and date(created_at) between date('" + input_start_date + "') and date('" + input_end_date + "')")
    temp_df = pd.DataFrame(sqlite_cursor.fetchall())
    print("temp_df.shape:", temp_df.shape)
    temp_df.columns = ['id', 'source', 'severity', 'message', 'created_at', 'duration_in_seconds']
    temp_df['created_at'] = pd.to_datetime(temp_df['created_at'])
    temp_df['duration_in_seconds'] = temp_df['duration_in_seconds'].astype(int)
    temp_df.sort_values(by=["created_at"], axis = 0, ascending = True, inplace = True, na_position ='last')

    shift_start_time_in_ist = "09:00 AM"
    shift_start_time_in_ist = shift_start_time_in_ist.upper()
    shift_start_time_in_ust_hour = int(shift_start_time_in_ist.split(":")[0]) - 5
    shift_start_time_in_ust_minute = int(shift_start_time_in_ist.split(":")[1].split(" ")[0]) - 30

    def calc_shift(r):
        event_date = r['created_at'].to_pydatetime()
        ist_shift_start_time_in_ust = event_date.replace(minute=0, hour=0, second=0, microsecond=0) + \
            timedelta(hours=shift_start_time_in_ust_hour, minutes=shift_start_time_in_ust_minute, seconds=0)
        ist_shift_end_time_in_ust = ist_shift_start_time_in_ust + timedelta(hours=12, minutes=0, seconds=0)
        # print("ist_shift_start_time_in_ust:", ist_shift_start_time_in_ust)
        # print("ist_shift_end_time_in_ust:", ist_shift_end_time_in_ust)
        if event_date >= ist_shift_start_time_in_ust and event_date <= ist_shift_end_time_in_ust:
            return event_date.strftime("%d-%b-%Y") + " IST"
        elif event_date < ist_shift_start_time_in_ust:
            prev_day = event_date.date() + timedelta(days=-1)
            return prev_day.strftime("%d-%b-%Y") + " PST"
        elif event_date > ist_shift_end_time_in_ust:
            return event_date.strftime("%d-%b-%Y") + " PST"

    # print(temp_df.head(5))
    temp_df = temp_df[(temp_df['duration_in_seconds'] >= 600) & (~temp_df['severity'].str.lower().isin(['info', 'warning']))]
    temp_df['shift_time'] = temp_df.apply(lambda x: calc_shift(x), axis=1)
    temp_df[['shift_day', 'shift']] = temp_df['shift_time'].str.split(' ', expand=True)

    print("Before delete::", temp_df.shape)
    # Delete duplicate events which occure more than once in a shift-day [like, pipeline not triggered in 12 hrs etc.]:
    temp_df['shift_date'] = pd.to_datetime(temp_df['shift_day'], format="%d-%b-%Y")
    temp_df['shift_date'] = temp_df['shift_date'].dt.date
    temp_df['rank'] = temp_df.groupby(['shift_date', 'shift', 'message'])['created_at'].rank(ascending=True)
    temp_df = temp_df[temp_df['rank'] < 2.0]
    # temp_df.drop(['shift_date', 'rank'], axis=1, inplace=True)
    print("After delete::", temp_df.shape)
    # print(temp_df.head(5))

    plt.rcParams["figure.figsize"] = (22, 8)
    temp_df.groupby('shift_date').id.count().plot(kind='bar')
    plt.title("Critical alert count between " + r"$\bf{" + input_start_date + "}$" + ' and ' + r"$\bf{" + input_end_date + "}$")
    plt.xlabel('')
    plt.ylabel(r"$\bf{" + 'Count' + "}$")
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.savefig('/Users/shubham.mallick/Downloads/foo.png', bbox_inches='tight')
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_total_critical_alerts = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()



    # Check if there are any critical message other than long running:
    df_critical_without_long_running_jobs = temp_df[~temp_df['message'].str.contains("SP is taking longer than expected")]
    # print("df_critical_without_long_running_jobs.shape:", df_critical_without_long_running_jobs.shape)
    df_critical_only_long_running_jobs = temp_df[temp_df['message'].str.contains("SP is taking longer than expected")]
    # print("df_critical_only_long_running_jobs.shape:", df_critical_only_long_running_jobs.shape)

    mapping_condition_without_long_running = {
        "proc_plugins_pub_date": "proc_plugins_pub_date",
        "kraken.remove_old_records": "kraken.remove_old_records, Lock timeout",
        "tvm_sla_team_daily total host count does not match that of tvm_sla_team_daily_no_grc": "host count tvm_sla_team_daily ≠ tvm_sla_team_daily_no_grc",
        "procedure_viper_gating()": "procedure_viper_gating",
        "CRITICAL: No new daily task being triggered in 12 hours": "No new daily task triggered 12 hours"
    }
    updated_msg = []
    for msg in df_critical_without_long_running_jobs['message']:
        found = False
        for k, v in mapping_condition_without_long_running.items():
            if k in msg:
                updated_msg.append(v)
                found = True
                break
        if not found:
            updated_msg.append(None)
    df_critical_without_long_running_jobs['updated_msg'] = updated_msg

    df_critical_only_long_running_jobs['proc_name'] = df_critical_only_long_running_jobs['message'].str.split(':WARNING:').str[0].str.lower()

    # plt.figure(figsize=(22, 6))
    fig, ax = plt.subplots(figsize=(18, 5))
    if len(df_critical_without_long_running_jobs) > 0:
        df_critical_without_long_running_jobs.groupby(['updated_msg']).size().sort_values(ascending=False).plot.barh()
    else:
        # Adding text inside a rectangular box by using the keyword 'bbox'
        # plt.text(0.5, 0.5, 'No data found in this date range', fontsize=25, bbox=dict(facecolor='pink', alpha=0.5))
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        ax.text(0.5 * (left + right), 0.5 * (bottom + top),
                'No data found in this date range',
                fontsize=25,
                bbox=dict(facecolor='pink', alpha=0.5),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    plt.grid(axis='x')
    plt.title("Critical [non-long running] alert types and their occurrences between " + r"$\bf{" + input_start_date + "}$" + ' and ' + r"$\bf{" + input_end_date + "}$")
    plt.xlabel("No. of Occurrences")
    plt.ylabel("Type of Alert")
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_critical_alerts_type_n_occurrence = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()



    # plt.figure(figsize=(22, 15))
    fig, ax = plt.subplots(figsize=(18, 5))
    if len(df_critical_only_long_running_jobs) > 0:
        df_critical_only_long_running_jobs.groupby(['proc_name']).size().sort_values(ascending=False).plot.barh()
    else:
        # Adding text inside a rectangular box by using the keyword 'bbox'
        # plt.text(0.5, 0.5, 'No long running jobs found in this date range', fontsize=25, bbox=dict(facecolor='pink', alpha=0.5))
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        ax.text(0.5 * (left + right), 0.5 * (bottom + top),
                'No data jobs found in this date range',
                fontsize=25,
                bbox=dict(facecolor='pink', alpha=0.5),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    plt.grid(axis='x')
    plt.title("Critical [long running] alert types and their occurrences between " + r"$\bf{" + input_start_date + "}$" + ' and ' + r"$\bf{" + input_end_date + "}$")
    plt.xlabel("No. of Occurrences")
    plt.ylabel("Proc Name")
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_critical_alerts_long_running = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()


    data = {
        'img_proc_daily_task_runtime': img_proc_daily_task_runtime,
        'img_total_critical_alerts': img_total_critical_alerts,
        'img_critical_alerts_type_n_occurrence': img_critical_alerts_type_n_occurrence,
        'img_critical_alerts_long_running': img_critical_alerts_long_running
    }
    return JsonResponse(data, safe=False)



def weekly_analytics(request):
    return render(request, 'weekly_analytics.html')


def get_weekly_analytics_data(request):
    if request.method == 'POST':
        input_week_start_date = request.POST['week_start_date'].strip()
        print("input_week_start_date:", input_week_start_date)

    week_start_date = datetime.strptime(input_week_start_date, "%Y-%m-%d")
    week_end_date = week_start_date + timedelta(days=7)
    input_week_end_date = week_end_date.strftime("%Y-%m-%d")

    sqlite_conn = sqlite3.connect(sqlite_db_file_path)
    sqlite_cursor = sqlite_conn.cursor()

    sqlite_cursor.execute("select * from proc_daily_tasks where date(created_at) between date('" + input_week_start_date + "') and date('" + input_week_end_date + "')")
    df_proc_daily_tasks = pd.DataFrame(sqlite_cursor.fetchall())
    print("df_proc_daily_tasks.shape:", df_proc_daily_tasks.shape)
    cols = [column[0] for column in sqlite_cursor.description]
    df_proc_daily_tasks.columns = cols
    df_proc_daily_tasks.sort_values(by=['created_at'], inplace=True, ascending=True)

    # Delete data if any row contains data of start task of previous day
    if "completed" in df_proc_daily_tasks['message'].iloc[0]:
        df_proc_daily_tasks = df_proc_daily_tasks[1:]
    if "started" in df_proc_daily_tasks['message'].iloc[-1]:
        df_proc_daily_tasks = df_proc_daily_tasks[:-1]
    # print(df_proc_daily_tasks.head(50))

    df_proc_daily_tasks = df_proc_daily_tasks.assign(idx=df_proc_daily_tasks.groupby('message').cumcount()).pivot(index='idx', columns='message', values='created_at').reset_index()
    cols = ['proc_daily_tasks started', 'proc_daily_tasks completed']
    df_proc_daily_tasks = df_proc_daily_tasks[cols]
    df_proc_daily_tasks[cols] = df_proc_daily_tasks[cols].apply(pd.to_datetime)
    # print(df_proc_daily_tasks.head(50))

    df_proc_daily_tasks['pipeline_time_taken'] = pd.to_datetime(df_proc_daily_tasks['proc_daily_tasks completed'].sub(df_proc_daily_tasks['proc_daily_tasks started']).dt.total_seconds(), unit='s').dt.time
    df_proc_daily_tasks = df_proc_daily_tasks.sort_values(by=['proc_daily_tasks started'], kind='mergesort', ascending=False)

    # display(df_proc_daily_tasks.head(5))
    temp_proc_daily_tasks = df_proc_daily_tasks[['proc_daily_tasks started', 'pipeline_time_taken']].copy()
    temp_proc_daily_tasks['proc_daily_tasks started'] = pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']).dt.date
    temp_proc_daily_tasks['pipeline_time_taken'] = pd.to_timedelta(pd.to_datetime(temp_proc_daily_tasks['pipeline_time_taken'].astype('string')).dt.time.astype('string'))
    temp_proc_daily_tasks['pipeline_time_taken_seconds'] = temp_proc_daily_tasks['pipeline_time_taken'].dt.total_seconds()
    # print(temp_proc_daily_tasks.head(50))
    temp_proc_daily_tasks = temp_proc_daily_tasks.groupby('proc_daily_tasks started', as_index=False).agg({'pipeline_time_taken_seconds':'mean'})
    temp_proc_daily_tasks = temp_proc_daily_tasks.sort_values(by=['proc_daily_tasks started'], kind='mergesort', ascending=False)
    temp_proc_daily_tasks['pipeline_time_taken_hour'] = temp_proc_daily_tasks['pipeline_time_taken_seconds']/3600
    temp_proc_daily_tasks = temp_proc_daily_tasks[(pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']) >= week_start_date) & (pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']) <= week_end_date)]
    # display(temp_proc_daily_tasks.head(20))

    fig, ax = plt.subplots(figsize=(15, 5))
    x = list(temp_proc_daily_tasks['proc_daily_tasks started'])
    y = list(temp_proc_daily_tasks['pipeline_time_taken_hour'])
    y = y[::-1]
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.scatter(range(len(x)), y)
    coef = np.polyfit(range(len(x)), y, 1)
    poly1d_fn = np.poly1d(coef)
    regression_points = poly1d_fn(range(len(x)))
    val_1 = regression_points[0]
    val_n = regression_points[-1]
    percentage_change = ((val_n - val_1) / val_1) * 100
    # display(HTML('<b>Percentage change:  <span style="color:red">' + str(round(percentage_change, 2)) + '</span></b>'))
    ax.plot(range(len(x)), y, regression_points, '--', label="Proc_Daily_Task Runtime")
    ax.axis('tight')
    plt.title('Proc_Daily_Task Runtime between Date ' + r"$\bf{" + week_start_date.strftime("%Y-%b-%d") + "}$" + ' and ' + r"$\bf{" + week_end_date.strftime("%Y-%b-%d") + "}$")
    plt.xlabel('Dates')
    plt.ylabel('Hours')
    plt.legend(loc='upper right')

    # Create names on the x axis
    x_ticks = sorted(list(set(list(pd.to_datetime(temp_proc_daily_tasks['proc_daily_tasks started']).dt.date))))
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.xticks(rotation = 90)
    plt.grid(axis='y')
    # plt.show()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_weekly_proc_daily_task_runtime = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()


    sqlite_cursor.execute("select * from messages_log where lower(severity) = 'critical' and date(created_at) between date('" + input_week_start_date + "') and date('" + input_week_end_date + "')")
    temp_df = pd.DataFrame(sqlite_cursor.fetchall())
    cols = [column[0] for column in sqlite_cursor.description]
    temp_df.columns = cols
    print("temp_df.shape:", temp_df.shape)
    print(temp_df.head(10))
    temp_df['created_at'] = pd.to_datetime(temp_df['created_at'])
    temp_df['duration_in_seconds'] = temp_df['duration_in_seconds'].astype(int)
    temp_df.sort_values(by=["created_at"], axis = 0, ascending = True, inplace = True, na_position ='last')
    
    shift_start_time_in_ist = "09:00 AM"
    shift_start_time_in_ist = shift_start_time_in_ist.upper()
    shift_start_time_in_ust_hour = int(shift_start_time_in_ist.split(":")[0]) - 5
    shift_start_time_in_ust_minute = int(shift_start_time_in_ist.split(":")[1].split(" ")[0]) - 30

    def calc_shift(r):
        event_date = r['created_at'].to_pydatetime()
        ist_shift_start_time_in_ust = event_date.replace(minute=0, hour=0, second=0, microsecond=0) + \
            timedelta(hours=shift_start_time_in_ust_hour, minutes=shift_start_time_in_ust_minute, seconds=0)
        ist_shift_end_time_in_ust = ist_shift_start_time_in_ust + timedelta(hours=12, minutes=0, seconds=0)
        # print("ist_shift_start_time_in_ust:", ist_shift_start_time_in_ust)
        # print("ist_shift_end_time_in_ust:", ist_shift_end_time_in_ust)
        if event_date >= ist_shift_start_time_in_ust and event_date <= ist_shift_end_time_in_ust:
            return event_date.strftime("%d-%b-%Y") + " IST"
        elif event_date < ist_shift_start_time_in_ust:
            prev_day = event_date.date() + timedelta(days=-1)
            return prev_day.strftime("%d-%b-%Y") + " PST"
        elif event_date > ist_shift_end_time_in_ust:
            return event_date.strftime("%d-%b-%Y") + " PST"

    temp_df['shift_time'] = temp_df.apply(lambda x: calc_shift(x), axis=1)
    temp_df[['shift_day', 'shift']] = temp_df['shift_time'].str.split(' ', expand=True)

    print("Before delete::", temp_df.shape)
    # Delete duplicate events which occure more than once in a shift-day [like, pipeline not triggered in 12 hrs etc.]:
    temp_df['shift_date'] = pd.to_datetime(temp_df['shift_day'], format="%d-%b-%Y")
    temp_df['shift_date'] = temp_df['shift_date'].dt.date
    temp_df = temp_df[(pd.to_datetime(temp_df['shift_date']) >= week_start_date) & (pd.to_datetime(temp_df['shift_date']) <= week_end_date)]
    print("After week filter::", temp_df.shape)
    temp_df['rank'] = temp_df.groupby(['shift_date', 'shift', 'message'])['created_at'].rank(ascending=True)
    temp_df = temp_df[temp_df['rank'] < 2.0]
    temp_df.drop(['shift_date', 'rank'], axis=1, inplace=True)
    print("After delete::", temp_df.shape)
    # display(temp_df.head(5))

    plt.rcParams["figure.figsize"] = (15, 6)
    temp_df.pivot_table(index=['shift_day'], columns='shift', aggfunc='size').plot(kind='bar')
    plt.title("Weekly critical event counts per shift, between " + r"$\bf{" + week_start_date.strftime("%Y-%b-%d") + "}$" + ' and ' + r"$\bf{" + week_end_date.strftime("%Y-%b-%d") + "}$")
    plt.xlabel('')
    plt.ylabel(r"$\bf{" + 'Count' + "}$")
    plt.grid(True)
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_weekly_critical_event_count_per_shift = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()

    # Check if there are any critical message other than long running:
    df_critical_without_long_running_jobs = temp_df[~temp_df['message'].str.contains("SP is taking longer than expected")].copy()
    print("df_critical_without_long_running_jobs.shape:", df_critical_without_long_running_jobs.shape)
    df_critical_only_long_running_jobs = temp_df[temp_df['message'].str.contains("SP is taking longer than expected")].copy()
    print("df_critical_only_long_running_jobs.shape:", df_critical_only_long_running_jobs.shape)

    mapping_condition_without_long_running = {
        "proc_plugins_pub_date": "proc_plugins_pub_date",
        "kraken.remove_old_records": "kraken.remove_old_records, Lock timeout",
        "tvm_sla_team_daily total host count does not match that of tvm_sla_team_daily_no_grc": "host count tvm_sla_team_daily ≠ tvm_sla_team_daily_no_grc",
        "procedure_viper_gating()": "procedure_viper_gating",
        "CRITICAL: No new daily task being triggered in 12 hours": "Pipeline not running"
    }
    updated_msg = []
    for msg in df_critical_without_long_running_jobs['message']:
        found = False
        for k, v in mapping_condition_without_long_running.items():
            if k in msg:
                updated_msg.append(v)
                found = True
                break
        if not found:
            updated_msg.append(None)
    df_critical_without_long_running_jobs['updated_msg'] = updated_msg

    df_critical_only_long_running_jobs['proc_name'] = df_critical_only_long_running_jobs['message'].str.split(':WARNING:').str[0].str.lower()
        
    plt.figure(figsize=(18, 5))
    df_critical_without_long_running_jobs.groupby(['updated_msg']).size().sort_values(ascending=False).plot.barh()
    plt.grid(axis='x')
    plt.title("Alert type vs. occurance between " + r"$\bf{" + week_start_date.strftime("%Y-%b-%d") + "}$" + ' and ' + r"$\bf{" + week_end_date.strftime("%Y-%b-%d") + "}$")
    plt.xlabel("No. of Occurance")
    plt.ylabel("Type of Alert")
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_weekly_critical_alerts_type_n_occurance = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()


    # plt.figure(figsize=(18, 5))
    fig, ax = plt.subplots(figsize=(18, 5))
    if len(df_critical_only_long_running_jobs) > 0:
        df_critical_only_long_running_jobs.groupby(['proc_name']).size().sort_values(ascending=False).plot.barh()
    else:
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        ax.text(0.5 * (left + right), 0.5 * (bottom + top),
                'No data found in this date range',
                fontsize=25,
                bbox=dict(facecolor='pink', alpha=0.5),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        # plt.text(0.5, 0.5, 'No long running jobs found in this date range', fontsize=25, bbox=dict(facecolor='pink', alpha=0.5))
    # # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(axis='x')
    plt.title("Long running Procedure vs. occurance between " + r"$\bf{" + week_start_date.strftime("%Y-%b-%d") + "}$" + ' and ' + r"$\bf{" + week_end_date.strftime("%Y-%b-%d") + "}$")
    plt.xlabel("No. of Occurance")
    plt.ylabel("Proc Name")
    # plt.show()
    fig = plt.gcf()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    strings = base64.b64encode(buf.read())
    img_weekly_critical_alerts_long_running = 'data:image/png;base64, ' + urllib.parse.quote(strings)
    plt.clf()
    plt.cla()
    plt.close()


    data = {
        'img_weekly_critical_event_count_per_shift': img_weekly_critical_event_count_per_shift,
        'img_weekly_proc_daily_task_runtime': img_weekly_proc_daily_task_runtime,
        'img_weekly_critical_alerts_type_n_occurance': img_weekly_critical_alerts_type_n_occurance,
        'img_weekly_critical_alerts_long_running': img_weekly_critical_alerts_long_running
    }
    return JsonResponse(data, safe=False)

