
import os
import shutil

os.system('wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EYF4Xt2V-rFEoLzbaUq6LVABByEe-Yc55-xFhIPhHkfK6A?e=LQc5Nq&download=1" -O RealSR_v2_ordered.tar')
shutil.unpack_archive('RealSR_v2_ordered.tar', 'datasets/.')
os.remove('RealSR_v2_ordered.tar')