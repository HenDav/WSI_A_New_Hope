import subprocess

subprocess.run(['python', 'train_reg.py',
                    '--test_fold', str(1),
                    '--epochs', str(2),
                    #'--dataset', 'LUNG',
                    '--dataset', 'Breast',
                    '--target', 'ER',
                    '--transform_type', 'bnfrs',
                    #'--transform_type', 'hedcfrs',
                    '--batch_size', str(5),
                    #'--bootstrap',
                    '--n_patches_test', '10',
                    '--n_patches_train', '10',
                    '--model', 'resnet50_3FC',
                    '--bootstrap'
                ])