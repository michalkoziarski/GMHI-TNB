import os


if __name__ == '__main__':
    for fold in range(10):
        command = f'sbatch run.sh train_ensemble_full.py -fold {fold}'

        os.system(command)
