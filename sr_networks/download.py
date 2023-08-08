
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
        os.system('wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EYF4Xt2V-rFEoLzbaUq6LVABByEe-Yc55-xFhIPhHkfK6A?e=LQc5Nq&download=1" -O RealSR_v2_ordered.tar')
        print('Extracting Data...')
        shutil.unpack_archive('RealSR_v2_ordered.tar', 'datasets/.')
        os.remove('RealSR_v2_ordered.tar')
    elif args.file == "RealSR_24_gen":
        print('downloading generated dataset from InterFlow...')
        os.system('wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EUey1T5fIvNEoh4qYX2eG2oB479zIHWEaNCLNTHkhX6dbA?e=sTY93D&download=1" -O interflow_2_4_images.tar')
        print('Extracting Data...')
        shutil.unpack_archive('interflow_2_4_images.tar', 'datasets/.')
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
        os.system('wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EbQdObqo_pNBoyfMjs10smcBakm9mbdMtFiviemceAaCdA?e=9dgaZ6&download=1" -O sr_networks.tar')
        print('Extracting Data...')
        shutil.unpack_archive('sr_networks.tar', 'pretrained_models/.')
        os.remove('sr_networks.tar')
    else:
        raise NotImplementedError()