
import os
import shutil
import argparse
import gdown

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, 
                        choices=["RealSR_234", "RealSR_24_gen", "DRealSR_3_test", "pretrained"], 
                        help='type one of "RealSR_234", "RealSR_24_gen", "DRealSR_3_test", "pretrained"')
    args = parser.parse_args()

    if args.file == "RealSR_234":
        print('downloading dataset...')
        os.system('wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0" "https://onedrive.live.com/download?resid=85CF5B7F538E2007%2143412&authkey=!APs3vr1pAFK7HGo" -O RealSR_v2_ordered.tar')
        print('Extracting Data...')
        shutil.unpack_archive('RealSR_v2_ordered.tar', 'datasets/.')
        os.remove('RealSR_v2_ordered.tar')
    elif args.file == "RealSR_24_gen":
        print('downloading generated dataset from InterFlow...')
        os.system('wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0" "https://onedrive.live.com/download?resid=85cf5b7f538e2007%2140332&authkey=!AEJDOr0suzDFk_o" -O interflow_2_4_images.tar')
        print('Extracting Data...')
        shutil.unpack_archive('interflow_2_4_images.tar', 'datasets/.')
        os.system('mv datasets/trainable_m1v_flow_scale24_final4 datasets/interflow_2_4_images')
        os.remove('interflow_2_4_images.tar')
    elif args.file == "DRealSR_3_test":
        print('downloading DRealSR x3 test...')
        drealsr3_test_id = '1sMwr6wSe_r2wxJ_h9OVoKCKyXd5NFbrQ'
        gdown.download(id=drealsr3_test_id, output='DRealsr_x3_test.zip', quiet=False)
        print('Extracting Data...')
        shutil.unpack_archive('DRealsr_x3_test.zip', 'datasets/.')
        os.rename('datasets/Test_x3', 'datasets/DRealsr_x3_test')
        os.remove('DRealsr_x3_test.zip')
    elif args.file == "pretrained":
        print('downloading pretrained sr_networks...')
        os.system('wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0" "https://onedrive.live.com/download?resid=85CF5B7F538E2007%2122525&authkey=!APKEwR_eJ8CLIsk" -O sr_networks.tar')
        print('Extracting Data...')
        shutil.unpack_archive('sr_networks.tar', 'pretrained_models/.')
        os.remove('sr_networks.tar')
    else:
        raise NotImplementedError()