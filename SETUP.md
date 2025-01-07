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
- tbd: dataset processing

# new model setup
- default model config is in ./config/default, create a copy named ./config/my_model_name
- edit config files as needed
- run create_new_model.py and enter my_model_name when prompted, your new model will be created in $MODELS_PATH/my_model_name

# model training
- the model creation process will generate a shell script for training any relevant modules in $MODELS_PATH/my_model_name/training/
- edit the training and accelerate config files as needed and then run the shell script to begin training

# sampling / testing
- run sample.py to launch an interactive local web ui
- sampling tests / presets are in $CONFIG_PATH/sampling
- the vae module can be sampled / tested by running src/tests/vae.py, config for the vae test is in $CONFIG_PATH/tests/vae_test.json