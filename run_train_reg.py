import subprocess

infer = False

if infer:
    subprocess.run(['python', 'inference_Multi_REG.py',
                    '--folds', '123456',
                    '--dataset', 'Breast',
                    '--num_tiles', '20',
                    '-ex', '38',
                    '--from_epoch', '1080'
                    ])
else:
    subprocess.run(['python', 'train_reg.py',
                    '--test_fold', str(2),
                    '--epochs', str(2),
                    #'--dataset', 'LUNG',
                    #'--dataset', 'Breast',
                    '--dataset', 'ABCTB',
                    '--target', 'ER',
                    #'--target', 'Her2',
                    #'--target', 'PDL1',
                    #'--transform_type', 'bnfrs',
                    #'--transform_type', 'hedcfrs',
                    '--batch_size', str(2),
                    '--n_patches_test', '10',
                    '--n_patches_train', '10',
                    '--model', 'PreActResNets.PreActResNet50_Ron()',
                    #'--model', 'nets.resnet50_with_3FC()',
                    '--bootstrap',
                    #'--transform_type', 'aug_receptornet',
                    '--transform_type', 'flip',
                    #'-fast',
                    '--mag', str(10),
                    '-time'
                    #'-im'
                    #'--balanced_sampling'
                ])

#train_reg.py --test_fold 1 --epochs 2 --dataset LUNG --target PDL1 --batch_size 5 --n_patches_test 10 --n_patches_train 10 --model resnet50_3FC --transform_type aug_receptornet