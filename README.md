## Improving Post-processing of Audio Event Detectors using Reinforcement Learning (IEEE Access)

Code for our paper titled _Improving Post-processing of Audio Event Detectors using Reinforcement Learning_, accepted in the IEEE Access journal.

### Installation

The code has been tested with _Python 3.8_, on _Ubuntu 20.04_ and _Windows 10_.

```shell
pip install -r requirements.txt
```

### Data preparation

Training data should be the raw class probability outputs of a Sound Event Detector, in ```.npy``` format. The class probability outputs should be in the _(0.0, 1.0)_ range and be 1 vector of length ```num_classes``` per frame.

An example for 5 classes:

_audio_file_0.npy_
```shell
frame 0: 0.4 0.1 0.3 0.2 0.8
frame 1: 0.2 0.6 0.9 0.7 0.3
...
```

Reference data should be one ```.txt``` file with the labels per audio file, in a format like the following example:

_audio_file_0.txt_

```shell
0.000	1.300	Speech
1.985	4.574	Running_water
4.574	6.676	Cat
9.160	9.776	Dog
```

### Training

After preparing training data and reference data, you can call the training script as follows:

```shell
python src/run.py --path_to_data "/path/to/training/data" --path_to_reference_data "/path/to/reference/data" --output_dir "/path/to/output/directory" --name "experiment_name"
```

### Citation

Please cite this work as:

```
@ARTICLE{9853543,
  author={Giannakopoulos, Petros and Pikrakis, Aggelos and Cotronis, Yannis},
  journal={IEEE Access}, 
  title={Improving Post-Processing of Audio Event Detectors Using Reinforcement Learning}, 
  year={2022},
  volume={10},
  number={},
  pages={84398-84404},
  doi={10.1109/ACCESS.2022.3197907}}
```
