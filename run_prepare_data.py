import subprocess

subprocess.run(['python', 'prepare_data.py',
                    '--segmentation', 'True',
                    #'--data_folder', r'C:\ran_data\TCGA',
                    #'--data_folder', r'C:\ran_data\TCGA_bad_examples_181020',
                    #'--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_bad_examples_181020',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Lung\Lung_1',
                    '--data_folder', r'C:\ran_data\gip-main_examples\Leukemia',
                    #'--data_folder', r'C:\ran_data\ABCTB',
                    #'--format','ABCTB'
                    '--format','MIRAX'
                    #'--format', 'TCGA'
                ])