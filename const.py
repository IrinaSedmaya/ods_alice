from pathlib import Path
ROOT_DIR = Path(__file__).resolve(strict=True).parent
# SRC_DIR = ROOT_DIR / 'src'
# TRAIN_DIR = ROOT_DIR / 'train'
# WEIGHTS_DIR = ROOT_DIR / 'weights'
DATA_DIR = ROOT_DIR / 'data'  # отступ и комментарии
TIMES = ['time' + str(i) for i in range(1, 11)]
SITES = ['site' + str(i) for i in range(1, 11)]
RANDOM_STATE = 33
