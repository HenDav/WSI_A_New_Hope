import numpy as np
import os
inference_files = {}

old = False
if old:
    # inference_files['exp38_epoch558_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_100.data'
    # inference_files['exp38_epoch558_test_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_10.data'
    # inference_files['exp36_epoch72_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp36\Inference\Model_Epoch_72-Folds_[1]-Tiles_100.data'
    # inference_files['test_2'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_2.data'
    '''inference_files['exp38_epoch482_test_3'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_3.data'
    inference_files['exp38_epoch482_test_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_10.data'
    inference_files['exp38_epoch482_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_100.data'
    inference_files['exp38_epoch482_test_200'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_200.data' '''
    # inference_files['exp38_epoch482_train_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[2, 3, 4, 5, 6]-Tiles_10.data'
    # inference_files['test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_100.data'
    # inference_files['test_500'] = 'Data from gipdeep/runs/79/Inference/Model_Epoch_12000-Folds_[2]-Tiles_500.data'
    # inference_files['test_1000'] = 'Data from gipdeep/runs/79/Inference/Model_Epoch_12000-Folds_[2]-Tiles_1000.data'
    '''inference_files['TCGA, mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp70\Inference\Model_Epoch_34-Folds_[1]-Tiles_500.data'
    inference_files['TCGA, no mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp63,64\exp63\Inference\Model_Epoch_48-Folds_[1]-Tiles_500.data'
    inference_files['TCGA dx only, mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp77\Inference\Model_Epoch_72-Folds_[1]-Tiles_500.data' '''
    # inference_files['ex38_epoch1080_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[1]-Tiles_500.data'
    # inference_files['ex38_epoch1080_train_fold2_20'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[2]-Tiles_20.data'
    # inference_files['ex38_epoch1060_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1060-Folds_[1]-Tiles_500.data'
    # inference_files['ex38_epoch1040_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1040-Folds_[1]-Tiles_500.data'
    # inference_files['ex38_epoch1020_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1020-Folds_[1]-Tiles_500.data'
    # inference_files['ex38_epoch1000_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1000-Folds_[1]-Tiles_500.data'
    # inference_files['ex38_epoch1080_test_500_single_infer'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[1]-Tiles_500_inference_REG.data'
    # inference_files['rons_epoch1607_test_500'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\Model_Epoch_1607-Folds_[1]-Tiles_500.data'
    # inference_files['rons_epoch1607_test_20_resnet_v2'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_070221.data'
    # inference_files['rons_epoch1607_test_20_resnet_v2_no_softmax'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_no_softmax_070221.data'
    # inference_files['rons_epoch1607_test_20_resnet_v2_no_softmax_fold2'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_no_softmax_fold2_070221.data'
    '''inference_files['rons_epoch1467_20_patches_fold1_test_mag10'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_mag10_resnet_v2_no_softmax_fold1_080221.data' 
    inference_files['rons_epoch1467_20_patches_fold1_test_mag20'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_mag20_resnet_v2_no_softmax_fold1_080221.data' '''
    '''inference_files['TCGA_ER_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
    inference_files['TCGA_ER_fold1_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1480-Folds_[1]-Tiles_500.data'
    inference_files['TCGA_ER_fold1_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1460-Folds_[1]-Tiles_500.data'
    inference_files['TCGA_ER_fold1_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1440-Folds_[1]-Tiles_500.data'
    inference_files['TCGA_ER_fold1_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1420-Folds_[1]-Tiles_500.data' '''

    '''inference_files['exp177_epoch740_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\our_slides_comparison_180321\Model_Epoch_740-Folds_[1]-Tiles_100.data'
    inference_files['exp177_epoch760_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\our_slides_comparison_180321\Model_Epoch_760-Folds_[1]-Tiles_100.data'
    inference_files['rons_epoch1467_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\inference\our_slides_comparison_180321\Model_Epoch_rons_model-Folds_[1]-Tiles_100.data' '''

    '''inference_files['exp226_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 226\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
    inference_files['exp239_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 239\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
    inference_files['exp240_fold2_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 240\Inference\Model_Epoch_1500-Folds_[2]-Tiles_500.data'
    inference_files['exp241_fold3_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 241\Inference\Model_Epoch_1500-Folds_[3]-Tiles_500.data'
    inference_files['exp242_fold4_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 242\Inference\Model_Epoch_1500-Folds_[4]-Tiles_500.data'
    inference_files['exp243_fold5_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 243\Inference\Model_Epoch_1500-Folds_[5]-Tiles_500.data' '''

    """inference_files['exp177_fold1_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_700-Folds_[1]-Tiles_500.data'
    inference_files['exp177_fold1_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_720-Folds_[1]-Tiles_500.data'
    inference_files['exp177_fold1_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_740-Folds_[1]-Tiles_500.data'
    inference_files['exp177_fold1_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_760-Folds_[1]-Tiles_500.data'
    inference_files['exp177_fold1_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_780-Folds_[1]-Tiles_500.data'
    inference_files['exp177_fold1_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_800-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp227_fold3_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_700-Folds_[3]-Tiles_500.data'
    inference_files['exp227_fold3_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_720-Folds_[3]-Tiles_500.data'
    inference_files['exp227_fold3_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_740-Folds_[3]-Tiles_500.data'
    inference_files['exp227_fold3_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_760-Folds_[3]-Tiles_500.data'
    inference_files['exp227_fold3_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_780-Folds_[3]-Tiles_500.data'
    inference_files['exp227_fold3_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_800-Folds_[3]-Tiles_500.data'"""

    """inference_files['exp203_fold2_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_700-Folds_[2]-Tiles_500.data'
    inference_files['exp203_fold2_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_720-Folds_[2]-Tiles_500.data'
    inference_files['exp203_fold2_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_740-Folds_[2]-Tiles_500.data'
    inference_files['exp203_fold2_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_760-Folds_[2]-Tiles_500.data'
    inference_files['exp203_fold2_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_780-Folds_[2]-Tiles_500.data'
    inference_files['exp203_fold2_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_800-Folds_[2]-Tiles_500.data'"""

    """inference_files['exp228_fold3_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1400-Folds_[3]-Tiles_500.data'
    inference_files['exp228_fold3_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1420-Folds_[3]-Tiles_500.data'
    inference_files['exp228_fold3_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1440-Folds_[3]-Tiles_500.data'
    inference_files['exp228_fold3_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1460-Folds_[3]-Tiles_500.data'
    inference_files['exp228_fold3_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1480-Folds_[3]-Tiles_500.data'
    inference_files['exp228_fold3_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1500-Folds_[3]-Tiles_500.data'"""

    """inference_files['exp229_fold1_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1400-Folds_[1]-Tiles_500.data'
    inference_files['exp229_fold1_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1420-Folds_[1]-Tiles_500.data'
    inference_files['exp229_fold1_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1440-Folds_[1]-Tiles_500.data'
    inference_files['exp229_fold1_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1460-Folds_[1]-Tiles_500.data'
    inference_files['exp229_fold1_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1480-Folds_[1]-Tiles_500.data'
    inference_files['exp229_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp230_fold2_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1400-Folds_[2]-Tiles_500.data'
    inference_files['exp230_fold2_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1420-Folds_[2]-Tiles_500.data'
    inference_files['exp230_fold2_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1440-Folds_[2]-Tiles_500.data'
    inference_files['exp230_fold2_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1460-Folds_[2]-Tiles_500.data'
    inference_files['exp230_fold2_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1480-Folds_[2]-Tiles_500.data'
    inference_files['exp230_fold2_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1500-Folds_[2]-Tiles_500.data'"""

    """inference_files['exp256_fold1_epoch1100_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1100-Folds_[1]-Tiles_500.data'
    inference_files['exp256_fold1_epoch1120_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1120-Folds_[1]-Tiles_500.data'
    inference_files['exp256_fold1_epoch1140_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1140-Folds_[1]-Tiles_500.data'
    inference_files['exp256_fold1_epoch1160_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1160-Folds_[1]-Tiles_500.data'
    inference_files['exp256_fold1_epoch1180_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1180-Folds_[1]-Tiles_500.data'
    inference_files['exp256_fold1_epoch1200_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp256\Inference\Model_Epoch_1200-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp263_fold1_epoch1200_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1200-Folds_[1]-Tiles_500.data'
    inference_files['exp263_fold1_epoch1220_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1220-Folds_[1]-Tiles_500.data'
    inference_files['exp263_fold1_epoch1240_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1240-Folds_[1]-Tiles_500.data'
    inference_files['exp263_fold1_epoch1260_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1260-Folds_[1]-Tiles_500.data'
    inference_files['exp263_fold1_epoch1280_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1280-Folds_[1]-Tiles_500.data'
    inference_files['exp263_fold1_epoch1300_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp263\Inference\Model_Epoch_1300-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp264_fold5_epoch600_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_600-Folds_[5]-Tiles_500.data'
    inference_files['exp264_fold5_epoch620_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_620-Folds_[5]-Tiles_500.data'
    inference_files['exp264_fold5_epoch640_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_640-Folds_[5]-Tiles_500.data'
    inference_files['exp264_fold5_epoch660_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_660-Folds_[5]-Tiles_500.data'
    inference_files['exp264_fold5_epoch680_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_680-Folds_[5]-Tiles_500.data'
    inference_files['exp264_fold5_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\lung_april21\exp264\Inference\Model_Epoch_700-Folds_[5]-Tiles_500.data'"""

    """inference_files['exp253_fold1_epoch1200_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1200-Folds_[1]-Tiles_500.data'
    inference_files['exp253_fold1_epoch1220_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1220-Folds_[1]-Tiles_500.data'
    inference_files['exp253_fold1_epoch1240_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1240-Folds_[1]-Tiles_500.data'
    inference_files['exp253_fold1_epoch1260_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1260-Folds_[1]-Tiles_500.data'
    inference_files['exp253_fold1_epoch1280_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1280-Folds_[1]-Tiles_500.data'
    inference_files['exp253_fold1_epoch1300_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp253\Inference\Model_Epoch_1300-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp258_fold2_epoch1200_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1200-Folds_[2]-Tiles_500.data'
    inference_files['exp258_fold2_epoch1220_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1220-Folds_[2]-Tiles_500.data'
    inference_files['exp258_fold2_epoch1240_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1240-Folds_[2]-Tiles_500.data'
    inference_files['exp258_fold2_epoch1260_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1260-Folds_[2]-Tiles_500.data'
    inference_files['exp258_fold2_epoch1280_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1280-Folds_[2]-Tiles_500.data'
    inference_files['exp258_fold2_epoch1300_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp258\Inference\Model_Epoch_1300-Folds_[2]-Tiles_500.data'"""

    """inference_files['exp260_fold3_epoch600_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_600-Folds_[3]-Tiles_500.data'
    inference_files['exp260_fold3_epoch620_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_620-Folds_[3]-Tiles_500.data'
    inference_files['exp260_fold3_epoch640_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_640-Folds_[3]-Tiles_500.data'
    inference_files['exp260_fold3_epoch660_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_660-Folds_[3]-Tiles_500.data'
    inference_files['exp260_fold3_epoch680_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_680-Folds_[3]-Tiles_500.data'
    inference_files['exp260_fold3_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp260\Inference\Model_Epoch_700-Folds_[3]-Tiles_500.data'"""

    """inference_files['exp279_fold1_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_800-Folds_[1]-Tiles_500.data'
    inference_files['exp279_fold1_epoch820_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_820-Folds_[1]-Tiles_500.data'
    inference_files['exp279_fold1_epoch840_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_840-Folds_[1]-Tiles_500.data'
    inference_files['exp279_fold1_epoch860_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_860-Folds_[1]-Tiles_500.data'
    inference_files['exp279_fold1_epoch880_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_880-Folds_[1]-Tiles_500.data'
    inference_files['exp279_fold1_epoch900_test_500'] = r'C:\Pathnet_results\MIL_general_try4\IHC_runs\exp279\Inference\Model_Epoch_900-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp277_fold1_epoch300_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_300-Folds_[1]-Tiles_500.data'
    inference_files['exp277_fold1_epoch320_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_320-Folds_[1]-Tiles_500.data'
    inference_files['exp277_fold1_epoch340_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_340-Folds_[1]-Tiles_500.data'
    inference_files['exp277_fold1_epoch360_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_360-Folds_[1]-Tiles_500.data'
    inference_files['exp277_fold1_epoch380_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_380-Folds_[1]-Tiles_500.data'
    inference_files['exp277_fold1_epoch400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\exp277\Inference\Model_Epoch_400-Folds_[1]-Tiles_500.data'"""

    """inference_files['exp283_fold2_epoch700_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_700-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch720_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_720-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch740_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_740-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch760_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_760-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch780_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_780-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch800_test_500_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_w_inky_slides\Model_Epoch_800-Folds_[2]-Tiles_500.data'"""

    """inference_files['exp283_fold2_epoch700_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_700-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch720_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_720-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch740_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_740-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch760_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_760-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch780_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_780-Folds_[2]-Tiles_500.data'
    inference_files['exp283_fold2_epoch800_test_500_no_inky_slides'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\exp283\Inference_no_inky_slides\Model_Epoch_800-Folds_[2]-Tiles_500.data'"""

    """inference_files['exp291_fold5_epoch900_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_900-Folds_[5]-Tiles_500.data'
    inference_files['exp291_fold5_epoch920_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_920-Folds_[5]-Tiles_500.data'
    inference_files['exp291_fold5_epoch940_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_940-Folds_[5]-Tiles_500.data'
    inference_files['exp291_fold5_epoch960_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_960-Folds_[5]-Tiles_500.data'
    inference_files['exp291_fold5_epoch980_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_980-Folds_[5]-Tiles_500.data'
    inference_files['exp291_fold5_epoch1000_test_500'] = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp291\Inference\Model_Epoch_1000-Folds_[5]-Tiles_500.data'"""

patient_level = True
save_csv = True

exp = '_o321'
fold = 5
target = 'PR'

#inference_dir = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp' + str(exp) +r'\Inference'
#inference_dir = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\ER\exp' + str(exp) +r'\Inference'
inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs', target, 'exp' + str(exp), 'Inference')
#epochs = [900, 920, 940, 960, 980, 1000]
if target == 'ER':
    epochs = list(np.arange(900,1001,20))
else:
    epochs = list(np.arange(400, 1001, 100))
#epochs = [1000] #temp

key_list = [''.join((target, '_fold', str(fold), '_exp', str(exp), '_epoch', str(epoch), '_test_500')) for epoch in epochs]
val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500.data')) for epoch in epochs]
#val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']', '-Tiles_500.data')) for epoch in epochs]
#key_list = [''.join(('exp', str(exp), '_fold', str(fold), '_epoch', str(epoch), '_test_500_herohe')) for epoch in epochs]
#val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']-Tiles_500_herohe.data')) for epoch in epochs]
inference_files = dict(zip(key_list, val_list))
inference_name = target + '_fold' + str(fold) + '_exp' + str(exp)
