import openslide
import cProfile
import sys
import numpy as np
import pstats


def my_read_region(slide):
    x = np.random.randint(50000)
    y = np.random.randint(50000)
    image = slide.read_region((x, y), 0,
                              (256, 256)).convert('RGB')

if sys.platform == 'win32':
    fn1 = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat\TCGA\TCGA-A2-A0CQ-01Z-00-DX1.4E5FB4E5-A08C-4C87-A3BE-0640A95AE649.svs'
    fn2 = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB\01-06-034.001.EX.B1.ndpi'
else:
    fn1 = r'/home/womer/project/All Data/TCGA/TCGA-D8-A1JA-01Z-00-DX1.BD43F94A-D5A8-490E-AED4-5F3AB24080FA.svs'
    fn2 = r'/home/rschley/All_Data/temp_ABCTB/temp_home_run_test/ABCTB/08-11-019.505.EX.3S.ndpi'

slide1 = openslide.OpenSlide(fn1)
slide2 = openslide.OpenSlide(fn2)

cProfile.run('my_read_region(slide1)', '../slide1_stats')

cProfile.run('my_read_region(slide2)', '../slide2_stats')

print('finished!')


p = pstats.Stats('slide1_stats')
p.strip_dirs().sort_stats(-1).print_stats()

p = pstats.Stats('slide2_stats')
p.strip_dirs().sort_stats(-1).print_stats()