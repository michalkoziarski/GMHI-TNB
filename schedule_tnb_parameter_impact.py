import os


if __name__ == '__main__':
    for n_estimators in [1, 10, 25, 50, 100, 250, 500, 1000]:
        for fold in range(10):
            command = f'sbatch run.sh train_tnb_parameter_impact.py -fold {fold} -n_estimators {n_estimators}'

            os.system(command)

    for max_features in [0.1, 0.25, 0.5, 0.75, 0.9]:
        for fold in range(10):
            command = f'sbatch run.sh train_tnb_parameter_impact.py -fold {fold} -max_features {max_features}'

            os.system(command)

    for max_samples in [0.1, 0.25, 0.5, 0.75, 0.9]:
        for fold in range(10):
            command = f'sbatch run.sh train_tnb_parameter_impact.py -fold {fold} -max_samples {max_samples}'

            os.system(command)
