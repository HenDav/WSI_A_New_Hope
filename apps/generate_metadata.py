# python peripherals
from pathlib import Path

# gipmed
from core.metadata import MetadataGenerator, MetadataGeneratorArgumentsParser

if __name__ == '__main__':
    metadata_generator_arguments = MetadataGeneratorArgumentsParser().parse_args()

    metadata_generator = MetadataGenerator(
        datasets_base_dir_path=metadata_generator_arguments.datasets_base_dir_path,
        tile_size=metadata_generator_arguments.tile_size,
        desired_magnification=metadata_generator_arguments.desired_magnification,
        metadata_file_path=metadata_generator_arguments.metadata_file_path,
        metadata_enhancement_dir_path=metadata_generator_arguments.metadata_enhancement_dir_path,
        log_file_path=metadata_generator_arguments.log_file_path,
        dataset_ids=metadata_generator_arguments.dataset_ids,
        minimal_tiles_count=metadata_generator_arguments.minimal_tiles_count)

    metadata_generator.save_metadata()

