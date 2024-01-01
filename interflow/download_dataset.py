
import os
import shutil

os.system('wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0" "https://onedrive.live.com/download?resid=85CF5B7F538E2007%2143412&authkey=!APs3vr1pAFK7HGo" -O RealSR_v2_ordered.tar')
shutil.unpack_archive('RealSR_v2_ordered.tar', 'datasets/.')
os.remove('RealSR_v2_ordered.tar')