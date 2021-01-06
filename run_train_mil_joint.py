import subprocess

'''subprocess.run(['python', 'train_mil_joint.py',
                '--test_fold', str(1),
                '--epochs', str(2),
                '--dataset', 'LUNG',
                '--target', 'PDL1',
                #'--transform_type', 'wcfrs',
                '--transform_type', 'bnfrs',
                #'--transform_type', 'hedcfrs',
                '--model', 'receptornet',
                '-im',
                '--tta'
                ])'''

'''subprocess.run(['python', 'train_mil_joint.py',
                '--test_fold', str(2),
                '--epochs', str(2),
                '--dataset', 'RedSquares',
                '--target', 'RedSquares',
                #'--transform_type', 'cbnfrsc',
                '--transform_type', 'pcbnfrsc',
                #'--transform_type', 'hedcfrs',
                '--model', 'receptornet',
                '-im',
                '--bootstrap',
                '--c_param','0.01'
                ])'''

#inference mil
subprocess.run(['python', 'inference_mil.py',
                #'--model_path',r'C:\Pathnet_results\MIL_general\exp19',
                #'--from_epoch', '157',
                #'--folds', '1',
                '--folds', '2',
                '--model_path',r'C:\Pathnet_results\MIL_general_try4\exp2',
                '--from_epoch', '200',
                '--dataset', 'LUNG',
                '--target', 'PDL1',
                '--model', 'receptornet',
                '--bootstrap',
                '--num_tiles', '100',
                '-ev'
                ])