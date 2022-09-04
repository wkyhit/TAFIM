import os

# Directories
# BASE_DIR = './'
BASE_DIR = '/content/TAFIM'

TF_BASE_DIR = os.path.join(BASE_DIR, 'tf_logs')

# Pretrained Checkpoints for manipulation models
pSp_ffhq_encode_pth = os.path.join(BASE_DIR, 'model_checkpoints', 'pSp', 'psp_ffhq_encode.pt')
# simswap_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'G_simswap.pth')
simswap_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap','people', 'latest_net_G.pth')
# simswap_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap','512', '550000_net_G.pth')

# simswap_arcface_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'arcface.pth')
simswap_arcface_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'arcface_checkpoint.tar')
STYLECLIP_BASE_DIR = os.path.join(BASE_DIR, 'model_checkpoints', 'StyleClip')

# Checkpoint paths for results
MANIPULATION_TESTS_BASE_DIR = os.path.join(BASE_DIR, 'tests')
CHECK_POINT_DIR = '/content/drive/MyDrive/adversial_attack/TAFIM_attack'
ATTACK_BASE_DIR = os.path.join(CHECK_POINT_DIR, 'attack_results')
PSP_ATTACK_BASE_DIR = os.path.join(ATTACK_BASE_DIR, 'pSp')
SIMSWAP_ATTACK_BASE_DIR = os.path.join(ATTACK_BASE_DIR, 'SimSwap')
STYLECLIP_ATTACK_BASE_DIR = os.path.join(ATTACK_BASE_DIR, 'StyleClip')
ALL_ATTACK_BASE_DIR = os.path.join(ATTACK_BASE_DIR, 'All')


dataset_paths = {
    'self_recon_train': '',
    'self_recon_val': '',
    'self_recon_test': '',
    'style_mix_src': '',
    'style_mix_tgt': '',
    # indicate the target and source image paths for face swap
    # target和source与原文SimSwap的定义相反（注意指定路径需要相反）
    'fs_train_src': '/content/TAFIM/dataset/train_target_2000',
    'fs_train_tgt': '/content/TAFIM/dataset/train_source_2000',
    'fs_val_src': '/content/TAFIM/dataset/val_target_100',
    'fs_val_tgt': '/content/TAFIM/dataset/val_source_100',
    'fs_test_src': '/content/TAFIM/dataset/val_target_100',
    'fs_test_tgt': '/content/TAFIM/dataset/val_source_100',
}