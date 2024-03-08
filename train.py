import bisect
import glob
import os
import re
import time

import torch
from dataset import CustomCOCODataset, COCODataset, collate_fn
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.nn as nn
from utils.gpu import collect_gpu_info
from utils.utils import save_ckpt

def train_one_epoch(model, optimizer, data_loader, device, epoch, args, scaler=None):
    # for p in optimizer.param_groups:
    #     p["lr"] = args.lr_epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    iters = len(data_loader) if args.iters < 0 else args.iters

    model.train()
    A = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
 

# -------------------------------------------------------------------------- #
def main(args):
    dataset = CustomCOCODataset(args.data_dir, args.json_file, transforms=T.Compose([
                                T.ToTensor(),]))
    d_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    device = "cuda"

    args.warmup_iters = max(1000, len(d_train))

    num_classes = 4 # including background class


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
    parser.add_argument("--data_dir", default="./val")
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