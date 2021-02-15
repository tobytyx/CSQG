# -*- coding: utf-8 -*-
import pymysql
import os
import json
MYSQL_USER = 'tyx'
MYSQL_PASS = 'tyx'
MYSQL_DB = 'graduation'


def get_conn():
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        charset='utf8mb4'
    )
    return conn
