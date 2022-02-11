import os

from utils import extract_tar

for f in os.listdir('data/raw'):
    if f.endswith('.tar.gz') or f.endswith('.tar'):
        extract_tar(os.path.join('data/raw',f),'data/raw')
        print(f'Extracted {f}')
