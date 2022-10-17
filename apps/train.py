# gipmed
from nn.experiments import Experiment, ExperimentArgumentsParser

if __name__ == '__main__':
    experiment_arguments = ExperimentArgumentsParser().parse_args()
    experiments = Experiment.from_json(json_file_path=experiment_arguments.json_file_path)
    for experiment in experiments:
        for model_trainer in experiment.model_trainers:
            model_trainer.plot_train_samples(batch_size=8, figsize=(40, 20), fontsize=50)
