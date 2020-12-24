import subprocess

'''subprocess.run(['python', 'train_mil_joint.py',
                '--test_fold', str(1),
                '--epochs', str(2),
                '--dataset', 'LUNG',
                '--target', 'PDL1',
                '--transform_type', 'wcfrs',
                #'--transform_type', 'hedcfrs',
                '--model', 'receptornet',
                '-im'
                ])'''

subprocess.run(['python', 'train_mil_joint.py',
                '--test_fold', str(2),
                '--epochs', str(2),
                '--dataset', 'RedSquares',
                '--target', 'RedSquares',
                '--transform_type', 'cbnfrsc',
                #'--transform_type', 'hedcfrs',
                '--model', 'receptornet',
                '-im',
                '--bootstrap'
                ])