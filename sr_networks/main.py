import train_inr
import train
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT-paths', type=str, nargs='+', default=['datasets/RealSR_v2_ordered/Nikon/Train_divided/2', 'datasets/RealSR_v2_ordered/Nikon/Train_divided/4'])
    parser.add_argument('--test-canon-GT-path', type=str, default='datasets/RealSR_v2_ordered/Canon/Test/3')
    parser.add_argument('--test-nikon-GT-path', type=str, default='datasets/RealSR_v2_ordered/Nikon/Test/3')
    parser.add_argument('--parameter-save-path', type=str, default='parameters/')
    parser.add_argument('--parameter-restore-path', type=str, default=None)
    parser.add_argument('--parameter-name', type=str)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--train-iteration', type=int)
    parser.add_argument('--validation-step', type=int, default=10000)
    parser.add_argument('--checkpoint-step', type=int, default=10000)
    parser.add_argument('--test-scale', type=float, default=3.)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, choices=["VDSR", "RCAN", "KPN", "MetaSR","HAN", "NLSN", "LIIF", "SwinIR"], 
                        help='type one of "VDSR" "RCAN" "KPN" "MetaSR" "HAN" "NLSN" "LIIF" "SwinIR"')
    parser.add_argument('--preload', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--test-data', type=str, choices=["RealSR", "DRealSR"], default="RealSR")
    
    # for only LIIF, MetaSR
    parser.add_argument('--encoder', type=str, choices=["EDSR_baseline", "RDN"], help='type one of "EDSR_baseline" "RDN"')
    args = parser.parse_args()
    print(f'[*] arguments : \n {args} \n\n')

    if args.model.find("LIIF") != -1 or args.model.find("Meta") != -1:  
        train_inr.test(args) if(args.test) else train_inr.train(args)  
    else:
        train.test(args) if(args.test) else train.train(args)
