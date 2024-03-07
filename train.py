import bisect
import glob
import os
import re
import time

import torch
from dataset import CustomCOCODataset, COCODataset
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.nn as nn
from engine import evaluate, train_one_epoch
from utils.gpu import collect_gpu_info
from utils.utils import save_ckpt

# -------------------------------------------------------------------------- #
def main(args):
    # dataset = CustomCOCODataset(args.img_dir, args.ann, transforms=T.Compose([
    #                             T.ToTensor(),]))
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    device = "cuda"
    dataset = COCODataset(args.data_dir, json_file=args.json_file, train=True)

    indices = torch.randperm(len(dataset)).tolist()
    d_train = torch.utils.data.Subset(dataset, indices)

    args.warmup_iters = max(1000, len(d_train))

    num_classes = max(d_train.dataset.classes) + 1 # including background class


    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes
    )
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=num_classes*4, bias=True)
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    print(model)
    model.to(device)

    # if args.distributed and args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

       
    # model = pmr.maskrcnn_resnet50(True, 3).to(device)
    
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A

        B = time.time()
        eval_output, iter_eval = evaluate(model, d_train, device, args)
        # eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))
    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_backbone", default=None, type=str, help="the backbone weights enum name to load")

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )

    parser.add_argument("--json_file", default="instances_val.json", help="coco format json file")
    parser.add_argument("--data_dir", default="./")
    parser.add_argument("--ckpt-path", default="Exp", help="path for checkpoint")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)