# SAMPLERNN-BASED NEURAL VOCODER FOR STATISTICAL PARAMETRIC SPEECH SYNTHESIS
Codes of the paper: 
Yang Ai, Hong-Chuan Wu, and Zhen-Hua Ling, "SAMPLERNN-BASED NEURAL VOCODER FOR STATISTICAL PARAMETRIC SPEECH SYNTHESIS," in Proc. ICASSP, 2018, pp. 5659–5663.

Usage:
First enter the root directory of the folder: `cd sampleRNN_tf`.

Data preparation:
Put the train, validiation and test waveforms (16kHz sample rate) into the corresponding folder in directory 'datasets/dataset/waveform',
and pt the train, validiation and test conditions into the corresponding folder in directory 'datasets/dataset/acoustic_condition'.

Traning and validiation:
If you want to train the model using single GPU, please see 'scripts/four_tier/train_valid.py'
If you want to train the model using multiple GPU, please see 'scripts/four_tier/train_valid_multiGPU.py'

Generation:
Please see 'scripts/four_tier/generation.py'

The beginning of these codes introduced the usage.
