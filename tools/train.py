# python tools/train.py
# import pdb;pdb.set_trace()

# 如果想对多类别目标检测进行修改，在下列脚本中搜索"20230904" (old: "20230816")
# tools/train.py : 修改 model.nc
# lib/models/YOLOP.py : 修改 self.nc
# lib/dataset/convert.py : 修改 id_dict_SDExpressway
# lib/core/evaluate.py : 修改 nc
# lib/core/function.py : 修改 nc、id_dict_SDExpressway
# tools/demo.py : 修改 id_dict_SDExpressway



import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode') # 多显卡设置
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify') # 多显卡设置
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold') # 置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS') # IOU阈值
    args = parser.parse_args()

    return args


def main():
    # set all the configurations # 设置所有配置
    args = parse_args() # 加载参数
    update_config(cfg, args) # 更新配置

    # Set DDP variables # 多显卡设置
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank # rank = -1
    #print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir # 设置日志

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank) # logger = <RootLogger root (INFO)> # final_output_dir = 'runs/BddDataset/_2023-07-08-10-17' # tb_log_dir = 'runs/BddDataset/_2023-07-08-10-17'

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args)) # 打印参数
        logger.info(cfg) # 打印配置

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        } # writer_dict = {'writer': <tensorboardX.writer...1d4d18640>, 'train_global_steps': 0, 'valid_global_steps': 0}
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model # 构建模型
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu') # device =device(type='cuda', index=0)

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")
    model = get_net(cfg).to(device) # model = MCnet(  (model): Sequential(    (0): Focus(      (conv): Conv(        (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)        (act): Hardswish()      )    )    (1): Conv(      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)      (act): Hardswish()    )    ……
    # print("load finished")
    #model = model.to(device)
    # print("finish build model")

    model_params = sum([param.nelement() for param in model.parameters()]) # 计算模型参数量
    print("Number of parameter: %.2fM" % ((model_params)/1e6))    

    # define loss function (criterion) and optimizer # 定义损失函数和优化器
    criterion = get_loss(cfg, device=device) # criterion = MultiHeadLoss(  (losses): ModuleList(    (0-2): 3 x BCEWithLogitsLoss()  ))
    optimizer = get_optimizer(cfg, model) # optimizer = Adam (Parameter Group 0    amsgrad: False    betas: (0.937, 0.999)    capturable: False    differentiable: False    eps: 1e-08    foreach: None    fused: None    lr: 0.001    maximize: False    weight_decay: 0)


    # load checkpoint model # 导入模型检测点 # 端到端训练或分步训练
    best_perf = 0.0
    best_model = False
    last_epoch = -1

    Encoder_para_idx = [str(i) for i in range(0, 17)] # Encoder_para_idx = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', ..., '16']
    Det_Head_para_idx = [str(i) for i in range(17, 25)] # Det_Head_para_idx = ['17', '18', '19', '20', '21', '22', '23', '24']
    Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)] # Da_Seg_Head_para_idx = ['25', '26', '27', '28', '29', '30', '31', '32', '33']
    Ll_Seg_Head_para_idx = [str(i) for i in range(34,43)] # Ll_Seg_Head_para_idx = ['34', '35', '36', '37', '38', '39', '40', '41', '42']

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine 余弦
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # lr_scheduler = <torch.optim.lr_scheduler.LambdaLR object at 0x7f0255615d00>
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH # begin_epoch = 0

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        ) # checkpoint_file = 'runs/BddDataset/checkpoint.pth'
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0,25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))
        
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        # model = model.to(device)

        if cfg.TRAIN.SEG_ONLY:  #Only train two segmentation branchs
            logger.info('freeze encoder and Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DET_ONLY:  #Only train detection branch
            logger.info('freeze encoder and two Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_SEG_ONLY:  # Only train encoder and two segmentation branchs
            logger.info('freeze Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers 
                if k.split(".")[1] in Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:    # Only train encoder and detection branchs
            logger.info('freeze two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False


        if cfg.TRAIN.LANE_ONLY: 
            logger.info('freeze encoder and Det head and Da_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DRIVABLE_ONLY:
            logger.info('freeze encoder and Det head and Ll_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False
        
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)


    # assign model params
    model.gr = 1.0
    # model.nc = 13 # 检测对象种类 20230816
    model.nc = 10 # 20230904
    print('bulid model finished')

    print("begin to load data") # 加载数据集
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ) # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 加载train数据集和val数据集
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
    # import pdb;pdb.set_trace()
    
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]: 
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('load data finished')
    
    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:
            logger.info("begin check anchors")
            run_anchor(logger,train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
        else:
            logger.info("anchors loaded successfully")
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            logger.info(str(det.anchors))

    # training # 开始训练
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    print('=> start training...')
    max_mIOU, max_IOU, max_mAP50 , min_Loss= 0, 0, 0, 1 # 标记可驾驶区域的最大mIOU，车道线的最大IOU，交通对象的最大mAP0.5，最小损失
    epoch_max_mIOU, epoch_max_IOU, epoch_max_mAP50, epoch_min_Loss = 0, 0, 0, 0 # 标记此时的epoch
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        epoch_best = False # 标记当前轮次有没有出现best指标
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
        
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            # print('validate')
            da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                epoch,cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank, model.nc
            )
            fi = fitness(np.array(detect_results).reshape(1, -1))  #目标检测评价指标

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
            logger.info(msg)

            # 打印最大的可驾驶区域mIOU、车道线IOU、交通对象mAP0.5，最小的损失Loss 对应的轮次和大小
            if min_Loss > total_loss:
                min_Loss = total_loss
                epoch_min_Loss = epoch
                epoch_best = True
            if max_mIOU < da_segment_results[2]:
                max_mIOU = da_segment_results[2]
                epoch_max_mIOU = epoch
                epoch_best = True
            if max_IOU < ll_segment_results[1]:
                max_IOU = ll_segment_results[1]
                epoch_max_IOU = epoch
                epoch_best = True
            if max_mAP50 < detect_results[2]:
                max_mAP50 = detect_results[2]
                epoch_max_mAP50 = epoch
                epoch_best = True

            msg_max = '\n'\
                                    'Min Loss: Epoch: [{epoch0}] Loss: {loss:.3f}\n'\
                                    'Max mIOU: Epoch: [{epoch1}] mIOU: {da_seg_miou:.3f}\n'\
                                    'Max IOU: Epoch: [{epoch2}] IOU: {ll_seg_iou:.3f}\n'\
                                    'Max mAP0.5: Epoch: [{epoch3}] mAP@0.5: {map50:.3f}'.format(
                                        epoch0 = epoch_min_Loss, loss = min_Loss,
                                        epoch1 = epoch_max_mIOU, da_seg_miou = max_mIOU, 
                                        epoch2 = epoch_max_IOU, ll_seg_iou = max_IOU,
                                        epoch3 = epoch_max_mAP50, map50 = max_mAP50)
            logger.info(msg_max)

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # save checkpoint model and best model
        # if rank in [-1, 0] and epoch%10 == 0:
            # 修改模型保存频率，每 10 epoches 保存一次，2023.07.14
        if rank in [-1, 0] and epoch_best:
            # 修改模型保存频率，遇到 epoch_best 保存一次，2023.09.08
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )

    # save final model # 保存模型
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()