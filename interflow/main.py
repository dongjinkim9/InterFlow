from train import *
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT-path', type=str, default='./datasets/RealSR_v2_ordered/Canon/Train_divided')
    parser.add_argument('--test-GT-path', type=str, default='./datasets/RealSR_v2_ordered/Nikon/Train_divided')
    parser.add_argument('--parameter-save-path', type=str, default='parameters/')
    parser.add_argument('--parameter-restore-path', type=str, default=None)
    parser.add_argument('--rrdb-restore-path', type=str, default=None)
    parser.add_argument('--parameter-name', type=str, default='test_model.pt')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=160)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--consistency-lambda', type=float, default=10)
    parser.add_argument('--ib-lambda', type=float, default=1)
    parser.add_argument('--mean-init', type=float, default=0.1)
    parser.add_argument('--std-init', type=float, default=0.1)
    parser.add_argument("--scales", type=int, nargs="+")
    parser.add_argument('--dequantization', action='store_true', default=True)
    parser.add_argument('--train-iteration', type=int, default=100000)
    parser.add_argument('--validation-step', type=int, default=1000)
    parser.add_argument('--save-validation-images', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()

    if(args.test):
        test(args)
    else:
        train(args)