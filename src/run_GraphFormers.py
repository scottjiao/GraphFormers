import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data_handler import DatasetForMatching, DataCollatorForMatching, SingleProcessDataLoader, \
    MultiProcessDataLoader
from src.models.modeling_graphformers import GraphFormersForNeighborPredict
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def cleanup():
    dist.destroy_process_group()


def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    config.neighbor_type = args.neighbor_type
    # model = GraphFormersForNeighborPredict(config)
    model = GraphFormersForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    return model


def train(local_rank, args, end, load):
    try:
        from src.utils import setuplogging
        setuplogging()
        os.environ["RANK"] = str(local_rank)
        setup(local_rank, args.world_size)
        if args.fp16:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        model = load_bert(args)
        logging.info('loading model: {}'.format(args.model_type))
        model = model.cuda()

        if load:
            model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
            logging.info('load ckpt:{}'.format(args.load_ckpt_name))

        if args.world_size > 1:
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            ddp_model = model

        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr}])

        data_collator = DataCollatorForMatching(mlm=args.mlm_loss, neighbor_num=args.neighbor_num,
                                                token_length=args.block_size)
        loss = 0.0
        global_step = 0
        best_acc, best_count = 0.0, 0
        for ep in range(args.epochs):
            start_time = time.time()
            ddp_model.train()
            dataset = DatasetForMatching(file_path=args.train_data_path)
            if args.world_size > 1:
                end.value = False
                dataloader = MultiProcessDataLoader(dataset,
                                                    batch_size=args.train_batch_size,
                                                    collate_fn=data_collator,
                                                    local_rank=local_rank,
                                                    world_size=args.world_size,
                                                    global_end=end)
            else:
                dataloader = SingleProcessDataLoader(dataset, batch_size=args.train_batch_size,
                                                     collate_fn=data_collator, blocking=True)
            for step, batch in enumerate(dataloader):
                if args.enable_gpu:
                    for k, v in batch.items():
                        if v is not None:
                            batch[k] = v.cuda(non_blocking=True)

                if args.fp16:
                    with autocast():
                        batch_loss = ddp_model(**batch)
                else:
                    batch_loss = ddp_model(**batch)
                loss += batch_loss.item()
                optimizer.zero_grad()
                if args.fp16:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()

                global_step += 1

                if local_rank == 0 and global_step % args.log_steps == 0:
                    logging.info(
                        '[{}] cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                            local_rank, time.time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                                        loss / args.log_steps))
                    loss = 0.0

                dist.barrier()
            logging.info("train time:{}".format(time.time() - start_time))

            if local_rank == 0:
                ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                logging.info("Star validation for epoch-{}".format(ep + 1))
                acc = test_single_process(model, args, "valid")
                logging.info("validation time:{}".format(time.time() - start_time))
                if acc > best_acc:
                    ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                    torch.save(model.state_dict(), ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")
                    best_acc = acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        start_time = time.time()
                        ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                        logging.info("Star testing for best")
                        acc = test_single_process(model, args, "test")
                        logging.info("test time:{}".format(time.time() - start_time))
                        exit()
            dist.barrier()

        if local_rank == 0:
            start_time = time.time()
            ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            logging.info("Star testing for best")
            acc = test_single_process(model, args, "test")
            logging.info("test time:{}".format(time.time() - start_time))
        dist.barrier()
        cleanup()
    except:
        import sys
        import traceback
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)


@torch.no_grad()
def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()

    data_collator = DataCollatorForMatching(mlm=args.mlm_loss, neighbor_num=args.neighbor_num,
                                            token_length=args.block_size)
    if mode == "valid":
        dataset = DatasetForMatching(file_path=args.valid_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=data_collator)
    elif mode == "test":
        dataset = DatasetForMatching(file_path=args.test_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        metrics = model.test(**batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1
    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("mode: {}, {}:{}".format(mode, key, metrics_total[key]))
    model.train()
    return metrics_total['main']


def test(args):
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    test_single_process(model, args, "test")
