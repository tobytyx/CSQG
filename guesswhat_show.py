# -*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
out_dir = "../guesswhat_new/out/games/"
# cls_file = os.path.join(out_dir, "cat_v_rcnn_cls.json")
# cls_3_file = os.path.join(out_dir, "5t_cat_v_rcnn_cls_3.json")
# cls_3_8t_file = os.path.join(out_dir, "8t_cat_v_rcnn_cls_3.json")
# hrnn_file = os.path.join(out_dir, "hrnn_v_rcnn.json")
# gt_file = os.path.join(out_dir, "cat_v_rcnn.json")
cls_3_punish = os.path.join(out_dir, "8t_cat_v_rcnn_cls_punish.json")
# with open(cls_file, mode="rb") as f:
#     cls_data = json.load(f)
with open(cls_3_punish, mode="rb") as f:
    punish_data = json.load(f)

categories = {
    'q_object': 0,
    'color': 1,
    'location': 2,
    'other': 3
}
i2c = {v:k for k,v in categories.items()}
L1 = []
L2 = []
L3 = []
for i in range(len(punish_data)):
    if (punish_data[i]["status"] == 'success') :
        L1.append(i)

j = 0

index = L1[j]
print("cls_3_punish: ", [i2c[q_cate.index(1)] for q_cate in punish_data[index]["q_cates"]])
print("cls_3_punish: ", punish_data[index]["qas"])

img_path = os.path.join("../guesswhat_new/data/imgs", punish_data[index]["image"])
j = j + 1
img = Image.open(img_path)
fig, ax1 = plt.subplots(1)
p = plt.imshow(img)

# CLS的是红色
bbox = punish_data[index]["objects"][punish_data[index]["guess_id"]]["bbox"]
rect_gt = patches.Rectangle((bbox["x"], bbox["y"]), bbox["width"], bbox["height"], linewidth=2, edgecolor='r', facecolor='none')
ax1.add_patch(rect_gt)
# 正确的是蓝色
bbox = punish_data[index]["object"]["bbox"]
rect_gt = patches.Rectangle((bbox["x"], bbox["y"]), bbox["width"], bbox["height"], linewidth=2, edgecolor='b', facecolor='none')
ax1.add_patch(rect_gt)
