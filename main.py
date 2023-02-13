import os
from pathlib import Path

import logging
import torch.multiprocessing as mp

from src.parameters import parse_args
from src.run import train, test
from src.utils import setuplogging

if __name__ == "__main__":

    setuplogging()
    gids=list(range(1))
    #gpus = ','.join([str(_ ) for _ in gids])
    gpus = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    args = parse_args()
    print(os.getcwd())
    args.log_steps = 5
    args.world_size = len(gids)  # GPU number
    #args.mode = 'train'
    #args.profile = "False"
    #args.model_type="CrossNodeGraphFormers"
    #args.model_type="GraphFormers"
    args.train_batch_size = 30
    args.neighbor_num = 5
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    logging.info(args)
    cont = False
    if args.mode == 'train':
        print('-----------train------------')
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            mp.spawn(train,
                     args=(args, end, cont),
                     nprocs=args.world_size,
                     join=True)
        else:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            #end = None
            train(0, args, end, cont)

    if args.mode == 'test':
        args.load_ckpt_name = "/data/zhou/graphformer/TuringModels/del/graphformers-dblp.pt"
        print('-------------test--------------')
        test(args)
