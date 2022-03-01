import os
import json

from munch import Munch
from torch.backends import cudnn
import torch
import random
import numpy as np
from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver

def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]



def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                                real=args.real,
                                                phm=args.phm,
                                                N=args.phm,
                                                which='source',
                                                img_size=args.img_size,
                                                batch_size=args.batch_size,
                                                prob=args.randcrop_prob,
                                                num_workers=args.num_workers),
                            ref=get_train_loader(root=args.train_img_dir,
                                                real=args.real,
                                                phm=args.phm,
                                                N=args.phm,
                                                which='reference',
                                                img_size=args.img_size,
                                                batch_size=args.batch_size,
                                                prob=args.randcrop_prob,
                                                num_workers=args.num_workers),
                            val=get_test_loader(root=args.val_img_dir,
                                                real=args.real,
                                                phm=args.phm,
                                                N=args.phm,
                                                img_size=args.img_size,
                                                batch_size=args.val_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            real=args.real,
                                            phm=args.phm,
                                            N=args.N,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            real=args.real,
                                            phm=args.phm,
                                            N=args.N,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    else:
        raise NotImplementedError


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

#device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.


if __name__ == '__main__':
    f = open('config.json',)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    args = Munch(data)

    args["layer_norm"] = False if args["layer_norm"] =="False" else True
    args["quat_inst_norm"]= False if args["quat_inst_norm"] =="False" else True
    args["quat_max_pool"]= False if args["quat_max_pool"] =="False" else True
    args["qsngan_layers"] = False if args["qsngan_layers"] =="False" else True
    args["htorch_layers"]= False if args["htorch_layers"] =="False" else True
    args["real"]= False if args["real"] =="False" else True
    args["phm"]= False if args["phm"] =="False" else True
    args["last_dense"]= False if args["last_dense"] =="False" else True
    device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    set_deterministic(args.seed)

    main(args)
