import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default="train",choices=['train', 'test'])
    parser.add_argument("--train_data_path",type=str,default="",)
    parser.add_argument("--train_batch_size",type=int,default=30)
    parser.add_argument("--valid_data_path",type=str,default="")
    parser.add_argument("--valid_batch_size", type=int, default=300)
    parser.add_argument("--test_data_path",type=str,default="")
    parser.add_argument("--test_batch_size", type=int, default=300)

    parser.add_argument("--model_dir", type=str, default='./ckpt')  # path to save
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)

    parser.add_argument("--warmup_lr", type=utils.str2bool, default=False)
    parser.add_argument("--savename", type=str, default='GraphFormers')
    parser.add_argument("--warmup_step", type=int, default=5000)
    parser.add_argument("--schedule_step", type=int, default=25000)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--neighbor_type", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--neighbor_num", type=int, default=5)
    parser.add_argument("--self_mask", type=utils.str2bool, default=False)



    # model training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=1000)
    parser.add_argument("--mlm_loss", type=utils.str2bool, default=False)
    parser.add_argument("--return_last_station_emb", type=utils.str2bool, default=False)
    parser.add_argument("--mapping_graph", type=utils.str2bool, default=False)

    # turing
    parser.add_argument("--model_type", default="GraphFormers", type=str)
    parser.add_argument("--model_name_or_path", default="./TuringModels/base-uncased.bin", type=str,
                        help="Path to pre-trained model or shortcut name. ")
    parser.add_argument("--config_name", default="./TuringModels/unilm2-base-uncased-config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")


    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default = 'ckpt/GraphFormers/GraphFormers-epoch-1.pt',
        help="choose which ckpt to load and test"
    )

    parser.add_argument("--pretrain_lr", type=float, default=1e-5)

    # half float
    parser.add_argument("--fp16", type=utils.str2bool, default=True)
    parser.add_argument("--neighbor_mask", type=utils.str2bool, default=False)
    parser.add_argument("--aggregation",type=str,default='mean',choices=['mean','max','gat'])

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()