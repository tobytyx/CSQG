# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import json
import copy
from data_provider.oracle_dataset import OracleDataset, oracle_collate
from models.oracle.baseline_model import OracleNetwork


class OracleWrapper(object):
    def __init__(self, data_dir, out_dir, model_name, tokenizer, device):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'imgs/')
        self.out_dir = os.path.join(out_dir, "oracle", model_name)
        self.tokenizer = tokenizer
        self.answer_dict = {0: 'Yes', 1: 'No', 2: 'N/A'}
        self.device = device
        self._load_model()
        self._load_dataset()

    def _load_model(self):
        with open(os.path.join(self.out_dir, "args.json")) as f:
            self.config = json.load(f)
        self.model = OracleNetwork(self.config, self.tokenizer, self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.out_dir, "params.pth.tar")))
        self.model.to(self.device)
        self.model.eval()

    def _load_dataset(self):
        self.dataset = OracleDataset(self.data_dir, None, self.config, self.tokenizer)

    def answer(self, game_data, visualize=False):
        games = copy.deepcopy(game_data)
        for game in games:
            game.questions = game.questions[-1]
            game.answers = 0
        self.dataset.load_data_from_games(games)
        batch = [self.dataset[i] for i in range(len(games))]
        batch = oracle_collate(batch)
        output = self.model(batch)
        output = torch.argmax(output, dim=-1)
        output = output.cpu().detach().tolist()
        for i in range(len(game_data)):
            answer = self.answer_dict[output[i]]
            game_data[i].answers.append(answer)
            if visualize:
                self.visualization(game_data[i])

    def visualization(self, game):
        filename = os.path.join(self.image_dir, game.img.filename)
        im = Image.open(filename).convert('RGB')
        w, h = im.size
        image = Image.new('RGB', (w, h + 60), (255, 255, 255))
        image.paste(im)
        bbox = game.object.bbox
        question = game.questions[-1]
        answer = game.answers[-1]

        # draw实体
        draw = ImageDraw.Draw(image)
        # font实体
        font = ImageFont.truetype('Arial.ttf', 20)
        draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
                   (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)))  # draw a square
        draw.text((20, h), question, font=font, fill="black")
        draw.text((w-50, h), answer, font=font, fill="red")
        del draw
        image.show()
