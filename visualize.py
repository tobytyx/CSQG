import os
import json
from PIL import Image, ImageDraw, ImageFont
from process_data.data_preprocess import Game, get_games
from utils.draw import mp_show
from process_data.tokenizer import GWTokenizer
from process_data.image_process import get_transform
from process_data.data_preprocess import get_games
data_dir = "./../data"
image_dir = data_dir + "/imgs"
dict_file = os.path.join(data_dir, "dict.json")
tokenizer = GWTokenizer(dict_file)
out_dir = "./../out/"
model_dir = "./models/"

transform = get_transform((224, 224))


def plot_game(game, model_name=None):
    print(game.id)
    filename = os.path.join(image_dir, game.img.filename)
    im = Image.open(filename).convert('RGB')
    qas = len(game.questions)
    w, h = im.size
    image = Image.new('RGB', (w, h+60+qas*30), (255, 255, 255))
    image.paste(im)
    # draw实体
    draw = ImageDraw.Draw(image)
    # font实体
    font = ImageFont.truetype('Arial.ttf', 30)
    # 圈出target object, 蓝色
    bbox = game.object.bbox
    draw.text((10, 10), text=str(game.id), fill="red", font=font)
    if model_name is not None:
        draw.text((w-250, 10), text=str(model_name), fill="red", font=font)
    draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
               (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)),fill="red", width=2)
    for o in game.objects:
        bbox = o.bbox
        draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
                   (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)), fill="blue", width=2)
    # 猜测错误时圈出guess object，红色
    if game.guess_id is not None:
        g_bbox = game.objects[game.guess_id].bbox
        draw.line(((g_bbox.x, g_bbox.y), (g_bbox.x + g_bbox.width, g_bbox.y),
                   (g_bbox.x + g_bbox.width,g_bbox.y + g_bbox.height),
                   (g_bbox.x, g_bbox.y + g_bbox.height), (g_bbox.x, g_bbox.y)), fill="blue", width=2)  # draw a square
    draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
               (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)), fill="red", width=2)
    # 打印出qa对
    for i, (q, a) in enumerate(zip(game.questions, game.answers)):
        draw.text((10, h+i*30), text=q, fill="black", font=font)
        draw.text((w-60, h+i*30), text=a, fill="red", font=font)
    # 输出最终game状态
    draw.text((10, h+qas*30), text=game.status, fill="red", font=font)
    del draw
    image.show()
    # image.save("tmp.jpg")
    # mp_show("tmp.jpg")


def load_json_file(filename):
    games = []
    with open(filename, 'r') as f:
        for line in f:
            game = json.loads(line.strip('\n'))
            g = Game(**game)
            games.append(g)
    return games

"""
loop_dir = os.path.join(out_dir, 'loop')
model1_file = os.path.join(loop_dir, "enc-dec-vgg.json")
model1_games = load_json_file(model1_file)
model3_file = os.path.join(loop_dir, "hred-rcnn.json")
model3_games = load_json_file(model3_file)
model7_file = os.path.join(loop_dir, "hred-va.json")
model7_games = load_json_file(model7_file)

for i in range(15, 20):
    if model1_games[i].__str__() != model7_games[i].__str__():
        plot_game(model1_games[i], model_name="Enc-Dec-VGG")
        plot_game(model3_games[i], model_name="Hred-RCNN")
        plot_game(model7_games[i], model_name="Hred-VA")
    else:
        print(model1_games[i].__str__())
        print(model7_games[i].__str__())
"""

games=get_games(data_dir, 'train')
for game in games:
    if game.img.id == 427135:
        plot_game(game)

games=get_games(data_dir, 'valid')
for game in games:
    if game.img.id == 427135:
        plot_game(game)

games=get_games(data_dir, 'test')
for game in games:
    if game.img.id == 427135:
        plot_game(game)