# -*- coding: utf-8 -*-
import os
import platform
import torch
import json
from PIL import Image, ImageDraw, ImageFont
from utils.constants import c_len

plats = ['cist-PowerEdge-R730', 'wangruonandeMacBook-Pro.local']


class QuestionWrapper(object):
    def __init__(self, data_dir, out_dir, model_name, tokenizer, device):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'imgs/')
        self.out_dir = os.path.join(out_dir, "qgen", model_name)
        self.tokenizer = tokenizer
        self.device = device
        self._load_model_dataset()

    def _load_model_dataset(self):
        with open(os.path.join(self.out_dir, "args.json")) as f:
            self.config = json.load(f)
        if self.config["model"] == "cat_base":
            from models.qgen.qgen_cat_base import QGenNetwork
            from data_provider.qgen_dataset import QuestionDataset, question_collate
        elif self.config["model"] == "cat_accu":
            from models.qgen.qgen_cat_accu import QGenNetwork
            from data_provider.qgen_dataset import QuestionDataset, question_collate
        elif self.config["model"] == "hrnn":
            from models.qgen.qgen_hrnn import QGenNetwork
            from data_provider.qgen_dataset import QuestionDataset, question_collate
        else:
            from models.qgen.qgen_baseline import QGenNetwork
            from data_provider.qgen_baseline_dataset import QuestionDataset, question_collate
        self.model = QGenNetwork(args=self.config, tokenizer=self.tokenizer, device=self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.out_dir, "params.pth.tar")), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.dataset = QuestionDataset(self.data_dir, None, self.config, self.tokenizer)
        self.question_collate = question_collate

    def question(self, game_data, turn, visualize=False):
        # 传入的game_data为list
        self.dataset.load_data_from_games(game_data)
        batch = [self.dataset[i] for i in range(len(game_data))]
        batch = self.question_collate(batch)
        tokens, labels = self.model.generate(batch)
        multi_cate = self.config.get("multi_cate", False)
        if "cat" in self.config["model"] and labels is not None:
            labels = labels.cpu().detach().tolist()
        for i in range(len(game_data)):
            if "cat" in self.config["model"] and labels is not None:
                if multi_cate:
                    label = labels[i]
                else:
                    label = [0] * c_len
                    label[labels[i]] = 1
                game_data[i].q_cates.append(label)
            else:
                label = [0] * c_len
                game_data[i].q_cates.append(label)
            question = self.tokenizer.decode(tokens[i])
            game_data[i].questions.append(question)
            if visualize:
                self.visualization(game_data[i], turn, None)

    def visualization(self, game, turn, tmp_att, topk=3):
        filename = os.path.join(self.image_dir, game.img.filename)
        im = Image.open(filename).convert('RGB')
        w, h = im.size
        image = Image.new('RGB', (w, h + 30*len(game.questions)), (255, 255, 255))
        image.paste(im)
        bbox = game.object.bbox
        img = self.dataset.image_builder.load_feature(game.img.filename)
        pos = img["bbox"]
        question = game.questions[-1]  # visualize last question

        # draw实体
        draw = ImageDraw.Draw(image)
        # font实体
        if platform.node() == plats[0]:
            # 服务器
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
        else:
            # 本地
            font = ImageFont.truetype('Arial.ttf', 20)
        # draw a square to highlight reffered object
        draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
                   (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)), fill="blue")

        for i, (q, a) in enumerate(zip(game.questions, game.answers)):
            draw.text((10, h + i * 30), text=q, fill="black", font=font)
            draw.text((w - 60, h + i * 30), text=a, fill="red", font=font)
        draw.text((10, h + (len(game.questions)-1) * 30), question, font=font, fill="black")
        if turn:
            # 有历史时，计算attention
            # 查看上一轮question对应的attention
            if isinstance(tmp_att, torch.Tensor):
                if tmp_att.sum(dim=1) != 1:
                    tmp_att = torch.softmax(tmp_att, dim=1)
                tmp_att = tmp_att.detach().cpu().numpy()
                tmp_att = tmp_att[0]
            if isinstance(game.attention, torch.Tensor):
                attention = game.attention.detach().cpu().numpy()
            else:
                attention = game.attention
            print(tmp_att)
            # 为了使用topk方法，将numpy先转为tensor
            tmp_att = torch.tensor(tmp_att)
            values, indices = torch.topk(tmp_att, topk)
            tmp_att = tmp_att.numpy()
            for i in range(topk):
                draw.rectangle(pos[indices[i]])
                draw.text((int(pos[indices[i]][0]), int(pos[indices[i]][1])),
                          "%.4f" % tmp_att[indices[i]], font=font, fill="black")
            # 查看累计注意力对应的attention
            print(attention[0])
            attention = torch.tensor(attention)
            values, indices = torch.topk(attention[0], topk)
            attention = attention.numpy()
            for i in range(topk):
                draw.rectangle(pos[indices[i]])
                draw.text((int(pos[indices[i]][0]), int(pos[indices[i]][1])),
                          "%.4f" % attention[0][indices[i]], font=font, fill="red")
        del draw
        image.show()
        image.save("{}_{}.jpg".format(game.id, turn))
        print("{}_{}.jpg".format(game.id, turn))
