import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, parse_devices, ensure_dir
from eval import SegEvaluator
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    all_dev = parse_devices('0')
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                'rgb_format': config.rgb_format,
                'gt_root': config.gt_root_folder,
                'gt_format': config.gt_format,
                'transform_gt': config.gt_transform,
                'x_root':config.x_root_folder,
                'x_format': config.x_format,
                'x_single_channel': config.x_is_single_channel,
                'class_names': config.class_names,
                'train_source': config.train_source,
                'eval_source': config.eval_source,
                'class_names': config.class_names}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)

    best_mean_IoU = 0
    best_weights_path = osp.join(config.checkpoint_dir, 'best.pth')

    hillfort_IoU_val = []
    background_IoU_val = []
    mean_IoU_val = []
    tloss = []

    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.metrics_dir)

    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        ''' if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)'''

    
        tloss.append((sum_loss / len(pbar)).item())

        with torch.no_grad():
            segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                     config.norm_std, network,
                                     config.eval_scale_array, config.eval_flip,
                                     all_dev, False, None,
                                     False)

            iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc = segmentor.run_for_each_epoch(model)

            if mean_IoU > best_mean_IoU:
                best_mean_IoU = mean_IoU
                engine.save_checkpoint(best_weights_path)


            n = iou.size
            for i in range(n):
                if config.class_names[i] == 'hillfort':
                    hillfort_IoU_val.append(iou[i])
                else:
                    background_IoU_val.append(iou[i])

            mean_IoU_val.append(mean_IoU)

            

        # Plot
        x = [*range(0, len(tloss), 1)]

        # Creating the plot for loss
        plt.plot(x, tloss, label='Training loss')
        plt.xticks(range(0, len(tloss), 5))
        # Adding labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # Save the plot
        plt.savefig(config.metrics_dir + '/loss.png')
        # Display the plot
        #plt.show()
        plt.close()

        x = [*range(0, len(mean_IoU_val), 1)]
        # Creating the plot for dice
        plt.plot(x, mean_IoU_val, label='Mean IoU')
        plt.plot(x, hillfort_IoU_val, label='Hillfort IoU')
        plt.plot(x, background_IoU_val, label='Background IoU')
        plt.xticks(range(0, len(mean_IoU_val), 5))
        # Adding labels and title
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        # Save the plot
        plt.savefig(config.metrics_dir + '/val.png')
        # Display the plot
        #plt.show()
        plt.close()


        tloss_file = open(config.metrics_dir + "/training_loss.txt", "w")
        miou_file = open(config.metrics_dir + "/mean_IoU.txt", "w")
        hillfort_file = open(config.metrics_dir + "/hillfort_IoU.txt", "w")
        background_file = open(config.metrics_dir + "/background_IoU.txt", "w")

        for e in tloss:
            tloss_file.write(f"{e}\n")
        for e in mean_IoU_val:
            miou_file.write(f"{e}\n")
        for e in hillfort_IoU_val:
            hillfort_file.write(f"{e}\n")
        for e in background_IoU_val:
            background_file.write(f"{e}\n")

        tloss_file.close()
        miou_file.close()
        hillfort_file.close()
        background_file.close()