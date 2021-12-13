# pairwise-margin-maximization

'''
python main.py --dataset cifar100 --model resnet --model-config "{'depth': 28, 'regime':'wide-resnet','width': [160, 320, 640]}" --epochs 200 --save resnet28_wide_cifar100_wt2_rmax_sqr_alpha_1e_5_1e_3 -b 64 --device-ids 0 --autoaugment --cutout --epochs 200 --optimize-mms --start-alpha-mms 1e-5 --end-alpha-mms 1e-3
'''
