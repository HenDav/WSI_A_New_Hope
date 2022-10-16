# gipmed
from nn.experiments import Experiment, ExperimentArgumentsParser

if __name__ == '__main__':
    experiment_arguments = ExperimentArgumentsParser().parse_args()
    experiment = Experiment.from_json(json_file_path=experiment_arguments.json_file_path)
    experiment.run()
