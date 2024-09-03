# setup notes - setup script is tbd

# environment setup
- install / update conda
- download / clone this repository
- create environment / activate environment
    - conda env create -f environment.yml
        - alternatively use conda.lock.yml for exact package versions
    - conda activate dualdiffusion
- copy .env.default to .env, edit as needed (local path config)

# dataset setup
- download / create dataset in the path specified in .env
    - the folder structure should be dataset_path/system_name/game_name/music.flac
- edit ./config/dataset/dataset.json as needed
- run process_dataset.py to setup dataset metadata

# new model setup
- default model config is in ./config/default, create a copy named ./config/my_model_name
- edit config files as needed
- run create_new_model.py and enter my_model_name when prompted, your new model will be created in ./models/my_model_name

# model training
- the model creation process will generate a shell script for training any relevant modules in ./models/my_model_name/training/
- edit the training and accelerate config files as needed and then run the shell script to begin training

# sampling
- sampling code is still being overhauled / cleaned up at the moment
- run sample.py to generate samples from a trained model
- sampling tests / presets are in ./config/sampling
- the vae module can be sampled / tested by running ./src/tests/vae.py, config for the vae test is in ./config/tests/vae_test.json