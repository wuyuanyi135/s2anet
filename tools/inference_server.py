from time import time

import mmcv
import torch
from aiohttp import web
import argparse
import numpy as np
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torch.utils.data import Dataset

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader
from mmdet.datasets.pipelines import Compose, Normalize, Pad, ImageToTensor, Collect, LoadImageFromFile
from mmdet.models import build_detector

import cv2
import logging

logging.basicConfig(level=logging.INFO)


class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.flag = np.array([1 for _ in data])  # required by group sampler

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def main(args):
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    distributed = False

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    classes_ = checkpoint['meta']['CLASSES']
    model.CLASSES = classes_
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pipeline = Compose([
        Normalize(**cfg.img_norm_cfg),
        Pad(size_divisor=32),
        ImageToTensor(keys=["img"]),
        Collect(keys=["img"])
    ])

    app = web.Application()
    routes = web.RouteTableDef()

    @routes.post("/inference")
    async def inference(request: web.Request):
        t_begin = time()
        post = await request.post()
        t_recv = time()
        data = []
        filenames = []
        shapes = []
        for key, body in post.items():
            img = cv2.imdecode(np.frombuffer(body.file.read(), np.uint8), 1)
            filenames.append(body.filename)
            shapes.append(img.shape)
            inputs = {}
            inputs['filename'] = body.filename
            inputs['img'] = img
            inputs['img_shape'] = img.shape
            inputs['ori_shape'] = img.shape
            inputs['scale_factor'] = 1.
            inputs['flip'] = False
            inputs = pipeline(inputs)
            # due to skipped MultiScaleFlipAug
            inputs['img_meta'] = [inputs['img_meta']]
            inputs['img'] = [inputs['img']]
            data.append(inputs)
        dataset = MemoryDataset(data)
        loader = build_dataloader(dataset, 1, 0, 1, False, shuffle=False)
        t_data = time()

        results = []
        for i, data in enumerate(loader):
            with torch.no_grad():
                results.append(model(return_loss=False, rescale=True, **data))
        t_inference = time()

        ret = {"results": {}}
        for i_file, (fn, categories) in enumerate(zip(filenames, results)):
            # initialize lists for caching different categories.
            rbbox = []
            scores = []
            cats = []
            for i, boxes in enumerate(categories):
                nbox = len(boxes[:, :-1].tolist())
                rbbox.extend(boxes[:, :-1].tolist())
                scores.extend(boxes[:, -1].tolist())
                cats.extend([i] * nbox)
            ret["results"][fn] = {
                "rbbox": rbbox,
                "scores":scores,
                "categories": cats,
                "shape": shapes[i_file]
            }
        t_final = time()

        ret["__time"] = __time = {
            "recv": t_recv - t_begin,
            "data": t_data - t_recv,
            "inference": t_inference - t_data,
            "total": t_final - t_begin,
        }

        ret["__classes"] = classes_

        out = f"Processed {len(filenames)} files: ["
        for shape in shapes:
            out += f"({shape[1]}x{shape[0]})"
        out += f"] Times: recv={__time['recv']:.2f}; data={__time['data']:.2f}; " \
               f"inference={__time['inference']:.2f}; total={__time['total']:.2f}"

        app.logger.info(out)

        return web.json_response(ret)

    app.add_routes(routes)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9292)
    parser.add_argument("config", help="path to config")
    parser.add_argument("checkpoint", help="path to checkpoint")
    args = parser.parse_args()
    main(args)
