import os

# Directories
BASE_DIR = './'
TF_BASE_DIR = os.path.join(BASE_DIR, 'tf_logs')

# Pretrained Checkpoints for manipulation models
pSp_ffhq_encode_pth = os.path.join(BASE_DIR, 'model_checkpoints', 'pSp', 'psp_ffhq_encode.pt')
# simswap_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'G_simswap.pth')
simswap_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap','512', '550000_net_G.pth')

# simswap_arcface_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'arcface.pth')
simswap_arcface_ckpt = os.path.join(BASE_DIR, 'model_checkpoints', 'SimSwap', 'arcface_checkpoint.tar')
STYLECLIP_BASE_DIR = os.path.join(BASE_DIR, 'model_checkpoints', 'StyleClip')

# Checkpoint paths for results
MANIPULATION_TESTS_BASE_DIR = os.path.join(BASE_DIR, 'tests')
ATTACK_BASE_DIR = os.path.join(BASE_DIR, 'attack_results')
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
    'fs_train_src': '',
    'fs_train_tgt': '',
    'fs_val_src': '',
    'fs_val_tgt': '',
    'fs_test_src': '/content/TAFIM/dataset/target_imgs_1000',
    'fs_test_tgt': '/content/TAFIM/dataset/source_imgs_1000',
}