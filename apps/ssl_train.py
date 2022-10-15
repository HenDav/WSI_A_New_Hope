# python core
import argparse

# gipmed
from nn.experiments import SSLExperimentArgumentsParser

if __name__ == '__main__':
    ssl_argument_parser = SSLExperimentArgumentsParser()
    ssl_experiment = ssl_argument_parser.create()
    ssl_experiment.run()
