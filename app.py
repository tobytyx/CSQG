# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, jsonify
import time
import os
import random
import shutil
import uuid
import pymysql
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from utils.online import guess, oracle, qgen
app = Flask(__name__)

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


with open("available_imgs.json") as f:
    available_imgs = json.load(f)


'''
    record_id: {
        "user_id": ""
        "last_time": time,
        "last_query": "",
        "history": [{"query": "", "answer": "", "category": ""}],
        "img_name": ""
    }
'''
control_state = {}

answer_dict = {
    "yes": "Yes",
    "no": "No",
    "n/a": "N/A"
}

answer_index_dict = {
    "yes": 0,
    "no": 1,
    "n/a": 2
}
conn = get_conn()


@app.route('/')
@app.route('/login/')
def login_method():
    return render_template("login.html")


@app.route('/check/user/', methods=["POST"])
def check_user_method():
    state, user_id = 0, "null"
    email = request.form["email"].strip()
    passwd = request.form["password"].strip()
    cur = conn.cursor()
    record_num = cur.execute(
        'select name, password from `user` where email=%s',
        args=str(email)
    )
    if record_num == 1:
        user_id, password = cur.fetchone()
        if passwd == password:
            return render_template("home.html", user_id=user_id)
    return render_template("login.html")


@app.route('/chat/', methods=["GET"])
def chat_method():
    user_id = request.args.get("user_id")
    return render_template("chat.html", user_id=user_id)


@app.route('/table/')
def tabel_method():
    user_id = request.args.get("user_id")
    return render_template("table.html", user_id=user_id)


@app.route('/home/', methods=["GET"])
def home_method():
    return render_template("home.html", user_id="sssn")


@app.route('/request/query/', methods=["POST"])
def request_query_method():
    record_id = request.form["record_id"].strip()
    user_id = request.form["user_id"].strip()
    img_name = request.form["img_name"].strip().split("/")[-1]
    global control_state
    last_time = time.time()
    if len(record_id) == 0:
        record_id = str(uuid.uuid4())
        control_state[record_id] = {
            "user_id": user_id,
            "last_time": last_time,
            "last_query": "",
            "history": [],
            "img_name": img_name,
            "guess": -1,
            "cache": {"turn": 0}
        }
        img_path = os.path.join("./static/gw_raw_imgs", img_name)
        tgt_path = os.path.join("./static/guesswhat_img", record_id)
        shutil.copy(img_path, tgt_path)
    else:
        control_state[record_id]["last_time"] = last_time
        answer = request.form["answer"].strip().lower()
        control_state[record_id]["history"].append({
            "query": control_state[record_id]["last_query"],
            "answer": answer_dict[answer]
        })
        oracle(control_state[record_id], answer_index_dict[answer])
    query = qgen(control_state[record_id])
    control_state[record_id]["last_query"] = query
    res = {
        "query": query,
        "record_id": record_id,
    }
    return jsonify(res)


@app.route('/request/img/', methods=["GET"])
def request_img_method():
    img_name = random.choice(list(available_imgs.keys()))
    img_path = "/static/gw_raw_imgs/" + img_name
    return jsonify({"img_path": img_path})


@app.route('/request/guess/', methods=["POST"])
def request_guess_method():
    record_id = request.form["record_id"].strip()
    global control_state
    last_time = time.time()
    control_state[record_id]["last_time"] = last_time
    answer = request.form["answer"].strip().lower()
    control_state[record_id]["history"].append({
        "query": control_state[record_id]["last_query"],
        "answer": answer_dict[answer]
    })
    oracle(control_state[record_id], answer_index_dict[answer])
    cats = available_imgs[control_state[record_id]["img_name"]]["cats"]
    spas = available_imgs[control_state[record_id]["img_name"]]["spas"]
    img_id = available_imgs[control_state[record_id]["img_name"]]["image_id"]
    predict_id = guess(control_state[record_id], img_id, cats, spas)
    bbox = available_imgs[control_state[record_id]["img_name"]]["bboxes"][predict_id]

    img_path = os.path.join("./static/gw_raw_imgs", control_state[record_id]["img_name"])
    tgt_path = os.path.join("./static/guesswhat_img", record_id)

    img = Image.open(img_path)
    _, ax1 = plt.subplots(1)
    plt.imshow(img)
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    rect_gt = patches.Rectangle((bbox["x"], bbox["y"]), bbox["width"], bbox["height"], linewidth=2, edgecolor='b', facecolor='none')
    ax1.add_patch(rect_gt)
    plt.savefig(tgt_path)
    if os.path.exists(tgt_path+".png"):
        shutil.move(tgt_path+".png", tgt_path)
    return jsonify({"img_path": "/static/guesswhat_img/" + record_id})


@app.route('/record/insert/', methods=["POST"])
def record_insert_method():
    record_id = request.form["record_id"].strip()
    guess = request.form["guess"].strip().lower()
    if guess == "yes":
        guess = 1
    else:
        guess = 0
    global control_state
    user_id = control_state[record_id]["user_id"]
    img_name = control_state[record_id]["img_name"]
    history = "\n".join(
        [each["query"] + "\t" + each["answer"] for each in control_state[record_id]["history"]]
    )
    print(control_state.pop(record_id))
    cur_str = 'insert into `guesswhat_record`'
    cur_str += '(`record_id`, `username`, `img_name`, `history`, `guess`) '
    cur_str += 'values(%s, %s, %s, %s, %s)'
    cur = conn.cursor()
    cur.execute(
        cur_str,
        args=(record_id, user_id, img_name, history, guess)
    )
    conn.commit()
    return jsonify({"guess": guess})


@app.route('/record/get/', methods=["GET"])
def get_record_method():
    user_id = request.args.get("user_id")
    cur = conn.cursor()
    cur.execute(
        'select record_id, img_name, create_time, guess from `guesswhat_record` where username=%s',
        args=str(user_id)
    )
    records = cur.fetchall()
    cur.close()
    records = [
        {
            "record_id": record[0],
            "img_name": record[1],
            "create_time": record[2],
            "guess": "yes" if record[3] == 1 else "no"
        } for record in records]
    return jsonify(records)


@app.route('/record/detail/', methods=["GET"])
def detail_record_method():
    record_id = request.args.get("record_id")
    cur = conn.cursor()
    cur.execute(
        'select history from `guesswhat_record` where record_id=%s',
        args=str(record_id)
    )
    records = cur.fetchone()[0]
    detail = [record.split("\t") for record in records.split("\n")]
    img_path = "/static/guesswhat_img/" + record_id
    return jsonify({"history": detail, "img_path": img_path})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9999")
