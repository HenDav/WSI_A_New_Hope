import subprocess

subprocess.run(['python', 'prepare_data.py',
                    '--segmentation', 'True',
                    '--data_folder', r'C:\ran_data\TCGA',
                    #'--data_folder', r'C:\ran_data\ABCTB',
                    #'--format','ABCTB'
                    '--format', 'TCGA'
                ])