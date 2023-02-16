import logging
import os
import random
import time
from collections import defaultdict
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data_handler import DatasetForMatching, DataCollatorForMatching, SingleProcessDataLoader, \
    MultiProcessDataLoader
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
import tensorboard


class blank_profile():
    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self,*args,**kwargs):
        return self
    def __exit__(self,*args,**kwargs):
        pass
    def step(self):
        pass


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    
def setup(rank, args):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def cleanup():
    dist.destroy_process_group()


def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    if args.model_type == "GraphFormers":
        from src.models.modeling_graphformers import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
    elif args.model_type == "GraphFormers_no_graph":
        from src.models.modeling_graphformers_with_no_graph import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
    elif args.model_type == "GraphFormers_no_rel_pos":
        from src.models.modeling_graphformers_with_no_rel_pos import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
    elif args.model_type == "GraphSageMax":
        from src.models.modeling_graphsage import GraphSageMaxForNeighborPredict
        model = GraphSageMaxForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == "CrossNodeGraphFormers":
        from src.models.modeling_graphformers_cross_node import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
        #model.load_state_dict(torch.load(args.model_name_or_path, map_location="cpu")['model_state_dict'], strict=False)
    return model


def train(local_rank, args, end, load):
    try:
        if local_rank == 0 and args.world_size > 1:
            from src.utils import setuplogging
            setuplogging()
        os.environ["RANK"] = str(local_rank)
        setup(local_rank, args)
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

        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.lr}])

        data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                                token_length=args.token_length, random_seed=args.random_seed)
        loss = 0.0
        global_step = 0
        best_acc, best_count = 0.0, 0
        
        start_structured_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        if args.profile=="True":
            profile_func=profile
        elif args.profile=="False":
            profile_func=blank_profile
        with profile_func(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./tmp/trace{start_structured_time}")
        ) as p:


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
                    """dataloader = MultiProcessDataLoader(dataset,
                                                        batch_size=args.train_batch_size,
                                                        collate_fn=data_collator,
                                                        local_rank=local_rank,
                                                        world_size=args.world_size,
                                                        global_end=end)"""
                local_step = 0

                data_time_start=    time.time()

                for step, batch in enumerate(dataloader):
                    data_time_end = time.time()
                    data_time = data_time_end - data_time_start
                    
                    with record_function("model_data_to_gpu"):
                        if args.enable_gpu:
                            for k, v in batch.items():
                                if v is not None:
                                    batch[k] = v.cuda(non_blocking=True)
                    data_time_gpu_end = time.time()
                    data_time_gpu = data_time_gpu_end - data_time_end
                    with record_function("model_inference"):
                        if args.fp16:
                            with autocast():
                                batch_loss = ddp_model(**batch)
                        else:
                            batch_loss = ddp_model(**batch)

                    with record_function("optimizer_step"):
                        loss += batch_loss.item()
                        optimizer.zero_grad()
                        if args.fp16:
                            scaler.scale(batch_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            batch_loss.backward()
                            optimizer.step()
                        local_step += 1
                        global_step += 1

                    p.step()
                    if local_rank == 0 and global_step % args.log_steps == 0:
                        #logging.info(
                        #    f"Epoch {ep}/{args.epochs} [{local_rank}] cost_time:{time.time() - start_time} step:{global_step}, lr:{optimizer.param_groups[0]['lr']}, train_loss: {loss / args.log_steps:.5f}")
                        
                        # print epoch, step, starting strctured time, now time, time used for this epoch, train loss
                        logging.info(
                            f"Ep:{ep} {step} start: { start_structured_time }| now: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}| meanBatchTime: {(time.time() - start_time)/local_step:.3f}s loss: {loss / args.log_steps:.3f} io: cpuData: {data_time*1000 :.1f}ms"
                        )
                        loss = 0.0

                    dist.barrier()
                    data_time_start = time.time()
                #logging.info(f"Epoch {ep}/{args.epochs} train time:{time.time() - start_time}")
                if local_rank == 0:
                    #ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                    #torch.save(model.state_dict(), ckpt_path)
                    #logging.info(f"Model saved to {ckpt_path}")

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
                        if best_count >= args.early_stop_duration:
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

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed)
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
