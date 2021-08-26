import subprocess

subprocess.run(['python', 'inference_heat_map.py',
                '--dataset', 'TCGA',
                '-ex', '321',
                '--from_epoch', '0',
                '-sn', 'TCGA-A2-A0CQ-01Z-00-DX1.4E5FB4E5-A08C-4C87-A3BE-0640A95AE649.svs'
                ])
