# Towards Realistic Single-Task Continuous Learning Research for NER

This repository holds the data and experimental code for the paper [Towards Realistic Single-Task Continuous Learning Research for NER](https://arxiv.org/abs/2110.14694).

The paper presents two different methods of splitting the [StackOverflowNER](https://github.com/jeniyat/StackOverflowNER) dataset into episodes. The first method splits the data into 5 temporal episodes, with each episode containing train/test examples within non-overlapping time periods. The second method (the "skewed" method) introduces class incrementality and data distribution shift.

## Data

All episodes used in the paper can be found in the `so_data` directory. The temporal episodes are located in `so_data/temporal_splits`, while the skewed episodes are located in `so_data/skewed_splits`.

The test episodes are in files named `so_temporal_test_{1-5}.json` or `so_test_{1-5}.json`, and we used the same test splits for all settings (non-CL baseline, CL w/o Replay, CL w/ Real Replay, and GDumb).

The non-CL baselines can be run by training on the files so_temporal_train_all_{1-5}.json for the temporal splits or so_train_all_{1-5}.json for the skewed splits. These training data files contain all 5 training episodes (for temporal or skewed splits) merged into a single file, then copied 5 times. You can thus use the same training procedure as the other settings, but the model will see all of the data in each episode.

The training files `so_train_{1-5}.json` and `so_temporal_train_{1-5}.json` can be used for the CL w/o Replay and CL w/ Real Replay settings.

The [GDumb](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_31) baseline maintains a fixed memory size at each episode, sampling to achieve a minimum level of diversity in the label set. We implemented the GDumb baseline by first creating the memory buffer at the end of each episode, then saving these pre-defined memory buffers in files. The contents of the memory buffer depend on the order in which examples are presented to the GDumb algorithm. Therefore we repeat the GDumb algorithm 10 times for each memory buffer size, with examples presented in a different random order each time. The `so_data/gdumb` directory contains these pre-defined memory buffers. All memory buffers created for the temporal episodes are in files starting with `gdumb_t`, while the memory buffers for skewed episodes are in files starting with simply `gdumb`. We then specify the maximum memory buffer size, the iteration id, and the episode number. To take an example, to run the first sample of the length 1000 memory buffer for the temporal data splits, you would train on `gdumb_t_1000_0_{1-5}.json`. 

## Creating New Episode Splits

New episodes can be created using the script `create_so_data.py`. You will need to clone the [StackOverflowNER](https://github.com/jeniyat/StackOverflowNER) repository in the same directory where this repository is located.

If you just want to re-create the temporal data splits, you can run `create_so_data.py --split_type temporal`. You can likewise run `create_so_data.py --split_type skewed` or `create_so_data.py --split_type both`. For the skewed splits, the default c value (controls the variance across episodes) is 5, but you can set it to 1 or 10 as well by passing `--c 1` or `--c 10` as arguments. If you wish to set c to other values, you will need to modify the `construct_skewed_dataset` function, since we specify different episode sizes for each c value to ensure we can construct episodes with the intended distributions.

If you want to fully re-create the temporal splits from scratch, you need to download the file Posts.xml from the [StackOverflow data dump](https://archive.org/details/stackexchange), and put it in the directory `so_data`. Then run `create_xml_dump.py` to create the directory `so_data/posts_texts` with posts labeled with the date they were created. You can then uncomment the line calling `load_xml_dump` in the `create_temporal_dataset` function in `create_so_data.py`, and comment out the lines below which load `from_xml_tokens` from the pickle.

You can rerun the memory buffer sampling for GDumb using the script `gdumb.py`. By default, this script will run GDumb for the temporal setting, then print out the json elements that need to be added to `settings.py` (see next section). You can pass `--split_type temporal` or `--split_type skewed` to change the dataset.

## Running Continual Learning Experiments

To run a continual learning experiment, you should first modify the `TASK_DICT` variable at the end of `settings.py`. You need to include all 5 episodes as separate entries. Entries are currently listed for skewed CL (`so_1`, etc), skewed non-CL baseline (`so_all_1`, etc), temporal CL (`so_t_1`, etc), and temporal non-CL baseline (`so_t_all_1`, etc).

You can then run, for example, `setupandrunexp.sh 0.25 0.2 5e-4 15 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels finetune exp_out/models .` to train the continual learning model on the skewed data with no replay with learning rate 5e-4, 15 epochs per episode. The first two arguments are parameters of [LAMOL](https://github.com/chho33/LAMOL), and we set them to be 0.25 and 0.2 respectively for all experiments. For convenience, there is a separate script for real replay, for example: `./setupandrunrealexp.sh 0.25 5e-4 15 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels exp_out/models .`.

Once a model has finished training on all 5 epochs, you should run the `runtest.sh` or `runtestreal.sh` scripts, for example:
`./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels finetune exp_out/models .`, which will test the 5 models saved after each training episode on all 5 test episodes. There is an analogous script `runtestreal.sh` for testing the CL model using real data replay. The model saved after training episode X (along with results for that model on all test episodes) is written out under `exp_out/models/gpt2/<CL type>/<sequence of episode names>/<training episode X>`.

Training and testing a baseline model (no CL) follows the same workflow as training the model with CL w/o replay, but you will specify different training episodes in `settings.py` and when calling the train and test scripts.

## Contact

Please contact Justin Payan (`jpayan@umass.edu`) for any questions/comments/discussion about the code or the paper. 