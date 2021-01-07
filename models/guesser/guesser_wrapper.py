# -*- coding: utf-8 -*-
import os
import json
from PIL import Image, ImageDraw
import torch
from data_provider.guesser_dataset import GuesserDataset, guesser_collate
from models.guesser.baseline_model import GuesserNetwork

from utils.draw import mp_show


class GuesserWrapper(object):
    def __init__(self, data_dir, out_dir, model_name, tokenizer, device):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'imgs/')
        self.out_dir = os.path.join(out_dir, "guesser", model_name)
        self.tokenizer = tokenizer
        self.device = device
        self._load_model()
        self._load_dataset()

    def _load_model(self):
        with open(os.path.join(self.out_dir, "args.json")) as f:
            self.config = json.load(f)
        self.model = GuesserNetwork(self.config, self.tokenizer, self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.out_dir, "params.pth.tar")))
        self.model.to(self.device)
        self.model.eval()

    def _load_dataset(self):
        self.dataset = GuesserDataset(self.data_dir, None, self.config, self.tokenizer)

    def guess(self, game_data, visualize=False):
        self.dataset.load_data_from_games(game_data)
        batch = [self.dataset[i] for i in range(len(game_data))]
        batch = guesser_collate(batch)
        output = self.model(batch)
        predict = torch.argmax(output, dim=1)
        target = batch[-1].detach().tolist()
        predict = predict.cpu().detach().tolist()
        if visualize:
            for game in game_data:
                self.visualization(game)
        success = []
        for i in range(len(game_data)):
            game_data[i].guess_id = int(predict[i])
            if predict[i] == target[i]:
                game_data[i].status = "success"
                success.append(1)
            else:
                game_data[i].status = "failure"
                success.append(0)
        return len(game_data), success

    def visualization(self, game):
        # for one game, visualize image and object box
        filename = os.path.join(self.image_dir, game.img.filename)
        im = Image.open(filename).convert('RGB')
        w, h = im.size
        image = Image.new('RGB', (w, h), (255, 255, 255))
        image.paste(im)
        # draw实体
        draw = ImageDraw.Draw(image)
        for i in range(len(game.objects)):
            bbox = game.objects[i].bbox
            draw.line(((bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height),
                       (bbox.x, bbox.y + bbox.height), (bbox.x, bbox.y)),fill=128)  # draw a square
            draw.text((bbox.x, bbox.y), text=str(i))
        del draw
        # image.show()
        image.save("tmp.jpg")
        mp_show("tmp.jpg")
