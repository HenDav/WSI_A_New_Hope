import subprocess

subprocess.run(['python', 'prepare_data.py',
                    '--segmentation',
                    #'--data_folder', r'C:\ran_data\TCGA',
                    '--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat',
                    #'--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_bad_examples_181020',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Lung\Lung_1',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Leukemia',
                    #'--data_folder', r'C:\ran_data\ABCTB',
                    '--grid'
                ])