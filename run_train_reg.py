import subprocess

infer = True

if infer:
    subprocess.run(['python', 'inference_REG.py',
                    '--folds', '123456',
                    '--dataset', 'TCGA',
                    '--num_tiles', '10',
                    '-ex', '38',
                    '--from_epoch', '482'
                    ])
else:
    subprocess.run(['python', 'train_reg.py',
                    '--test_fold', str(1),
                    '--epochs', str(2),
                    #'--dataset', 'LUNG',
                    '--dataset', 'Breast',
                    #'--target', 'ER',
                    '--target', 'Her2',
                    #'--target', 'PDL1',
                    #'--transform_type', 'bnfrs',
                    #'--transform_type', 'hedcfrs',
                    '--batch_size', str(5),
                    '--n_patches_test', '10',
                    '--n_patches_train', '10',
                    '--model', 'resnet50_3FC',
                    '--bootstrap',
                    '--transform_type', 'aug_receptornet',
                    #'-im'
                    '--balanced_sampling'
                ])

#train_reg.py --test_fold 1 --epochs 2 --dataset LUNG --target PDL1 --batch_size 5 --n_patches_test 10 --n_patches_train 10 --model resnet50_3FC --transform_type aug_receptornet