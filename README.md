# Emotion-Aware-AI
This research project aims to increase user engagement of learning games with a topic in computer science by dynamically adjusting the difficulty level through an emotion-aware NPC (Non-player character). The NPC, powered by an facial emotion recognition (FER) algorithm that categorizes facial expressions into 8 emotion types (including 3 custom ones), detects the current state of players via a combination of camera feed and in-game text sent between players.

## Data
Following the standard of well-known dataset FER-2013 created by Pierre Luc Carrier and Aaron Courville, integer labels are used to classify facial images expressing eight emotions, three of which (in italics) are tailored to the needs of this project:
- 0: Angry -- 4953
- 1: *Frustration* -- 265
- 2: *Boredom* -- 238
- ~~2: *Distracted*~~
- 3: Happy -- 8989
- 4: Sad -- 6077
- 5: Surprise -- 4002
- 6: Neutral -- 6198
> [!Note]
> *Distracted* is temporarily removed from our training dataset due to a failure in collecting meaningful images expressing this particular emotion. It may be added back in future if we're able to gather relevant images from other sources. Class labels are adjusted accordingly.

### How images are collected for custom emotions
Facial images representing frustration, distracted and boredom are fetched from Google Images using API calls. To ensure the diversity of our images, a basic search string is combined with other keywords like "men", "women" etc. before being passed to an API call. Faces from returned results are then extracted and cropped to a standard size of 48x48 with OpenCV, followed by a conversion to pixel string before being added to a csv file.

Finally, we manually inspect every pixel string to remove incorectly detected faces and dupliate images. This process is performed in each API call that collects 50 images and again every two API calls when a total of 100 images are fetched. 

### How image labels are assigned
We intentionally assigned our custom emotions to the exact two that are removed from the original FER2013 datasets -- disgust and fear, closely aligning our dataset to established practices.

### How to run our image collection program
Since custom images are collected via Google Images API, make sure to load your google API key and custom search engine ID into a .env file in the directory where main.py is located. See Google doc [here](https://developers.google.com/custom-search/v1/overview). Then, simply run `python main.py` in [data](data/main.py) and choose the first option on a menu that pops up.
> [!Note]
> Make sure to comment out the code in `main.py` that checks for `.env` credentials if you don't intend to run this program for image collection. 

> [!CAUTION]
> Other modes in the menu may break since they have been deprecated upon finishing data collection. Please see comments in [main.py](data/main.py). 

## FER Model
We built and experimented with two state-of-the-art [facial emotion recognition(FER)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=resemotenet-bridging-accuracy-and-loss) models, [ResEmoteNet](https://arxiv.org/pdf/2409.10545) and [EmoNeXt](https://ieeexplore.ieee.org/abstract/document/10337732), based on their original implementation, with a particular focus on the first due to various research constraints. 

However, it quickly caught my attention that ResEmoteNet performs far worse than the 79.79% test accuracy that the authors claim to have achieved when training with the same FER2013 benchmark dataset. Furthmore, a substantial amount of overfitting was observed when training with the specified hyperparameters from the original paper. These results are illustrated in following graphs

<img width="50%" alt="val loss" src="asset/val_loss.png" style="display: inline-block; margin-right: 10px;" /><img width="50%" alt="test loss" src="asset/test_loss.png" style="display: inline-block;" />
<img width="50%" alt="val acc" src="asset/val_acc.png" style="display: inline-block; margin-right: 10px;" /><img width="50%" alt="test acc" src="asset/test_acc.png" style="display: inline-block;" />

To combat these problems, we explored the following techniques:
- **Spatial Transformation Network(STN)**: strengthen geometric invariance of a neural network model by learning to perform spatial transformation on input images that eliminate translation, rotation and warping.
- **Data augmentation**: enhance existing training data transformation pipeline by adding more data augmentations, such as RandomRotation, RandomAffine etc.;
- **Weighted loss**: scale up cross entropy loss on minority classes and scale down on majority classes;
- **Weighted sampling**: add a `WeightedRandomSampler` to training dataloader to balance sampling probabilities between majority classes and minority classes.

## Model Training

### Environment setup
We recommend building a virtual environment on your local machine to run our scripts. 
1. Launch a virtual enviroment(for example, pipenv).
```
pipenv shell
```

2. Clone the repository.
```
git clone https://github.com/Cherisea/Emotion-Aware-AI.git
```

3. Install required libraries.
```
pipenv install -r requirements.txt
```

### Data setup

1. Create a symbolic link to your data folder:
```bash
# From the project root directory
ln -s /path/to/your/data/folder data
```

2. Alternatively, you can manually copy your data:
```bash
# From the project root directory
cp -r /path/to/your/data/folder data
```

The data folder should contain the following structure:
```
data/
├── fer2013/             # Your dataset folder
│   ├── train/            # Training subfolder
│       ├── train_label/     # Training labels
│       └── image1.jpg       # Training images
│       └── image2.jpg       # Training images
│       └── ...
│   ├── val/              # Validation subfolder
│       ├── val_label/       # Validation labels
│       └── image1.jpg       # Validation images
│       └── image2.jpg       # Validation images
│       └── ...
│   └── test/              # Test subfolder
│       ├── test_label/      # Test labels
│       └── image1.jpg       # Test images
│       └── image2.jpg       # Test images
│       └── ...
└── main.py               # Data collection script
```

> [!Note]
> Make sure your data folder contains the required subdirectories (train, val, test) with the appropriate emotion-labeled subfolders.

### Environment variables

Create a `.env` file in the root directory with the following variables for logging to wandb:
```
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
```

### Training models
If training on local machines, use the following command:

```bash
python train_resnet.py --dataset_path <path_to_dataset> --output_dir <output_directory> --epochs <number_of_epochs> --batch_size <batch_size> --lr <learning_rate> --momentum <momentum> --weight_decay <weight_decay> --early_stop <early_stopping_patience> --num_workers <num_workers>
```

Example:
```bash
python train_resnet.py --dataset_path fer2013 --output_dir /result --epochs 200 --batch_size 16 --lr 0.001 --momentum 0.9 --weight_decay 1e-3 --early_stop 10 --num_workers 4
```
Command for training EmoNeXt is similar. Consult our code for more details. If training on a HPC like Discovery, move to the next section [Docker Setup](#docker-setup). 


## Docker setup

### Building the docker image

1. Make sure you have Docker installed on your system.

2. Build the Docker image using the provided Dockerfile:
```bash
docker build -t [docker_tag] .
```

3. Push the image to a container registry:
```bash
docker push [docker_tag]
```

### Docker configuration

The Dockerfile is configured to:
- Use Python 3.9 as the base image
- Install all required dependencies from requirements.txt
- Set up the working directory as /app
- Make the result directory writable
- Use bash as the entry point

### Docker ignore

The `.dockerignore` file excludes:
- Git-related files
- Python cache files
- Virtual environments
- Test files
- Asset directories
- Data files
- Result directories
- Wandb logs
- Documentation files

## Project structure

```
Emotion-Aware-AI/
├── data/               # Dataset directory
├── models/             # Model implementations
│   ├── Resnet.py        # ResEmoteNet model
│   └── Emonext.py       # EmoNeXt model
├── train_resnet.py     # ResEmoteNet training script
├── train_emonet.py     # EmoNeXt training script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── .dockerignore       # Docker ignore rules
└── README.md           # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.



