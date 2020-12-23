import subprocess

subprocess.run(['python', 'prepare_data.py',
                    '--segmentation',
                    #'--data_collection',
                    '--grid',
                    #'--stats',
                    #'--data_folder', r'C:\ran_data\TCGA',
                    #'--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat',
                    #'--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_bad_examples_181020',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Lung',
                    #'--data_root', r'C:\ran_data\Lung_examples',
                    '--data_root', r'C:\ran_data\RedSquares',
                    #'--data_root', r'C:\ran_data\HEROHE_examples',
                    #'--dataset','HEROHE',
                    '--dataset','RedSquares',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Leukemia',
                    #'--data_folder', r'C:\ran_data\ABCTB',
                ])