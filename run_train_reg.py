import subprocess

subprocess.run(['python', 'train_reg.py',
                    '--test_fold', str(1),
                    '--epochs', str(2),
                    '--dataset', 'LUNG',
                    '--target', 'PDL1',
                    '--transform_type', 'wcfrs',
                    #'--transform_type', 'hedcfrs',
                    '--batch_size', str(5),
                    #'--bootstrap',
                    '--n_patches_test', '10',
                    '--n_patches_train', '10',
                    '--model', 'resnet50_3FC'
                ])