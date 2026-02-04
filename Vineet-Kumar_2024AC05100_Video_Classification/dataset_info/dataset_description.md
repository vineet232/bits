# Dataset Description

This assignment project uses a customized subset of the UCF-101 action recognition dataset, obtained from Kaggle, for the task of human action classification.

Dataset URL:
https://www.kaggle.com/datasets/aisuko/ucf101-subset

The dataset contains short video clips belonging to many categories.
However, in ths case following three action categories are chosen:

- PullUps  
- Punch  
- PushUps  

These actions were selected because they involve distinct upper-body movements, yet share certain visual and motion similarities. This makes the dataset suitable for evaluating both appearance-based features (such as color, texture, and shape) and motion-based representations.

The videos vary in background conditions, execution styles, camera viewpoints, and motion speed. As a result, the dataset reflects several real-world challenges, including intra-class variation, partial occlusions, and subtle differences between actions.

For this assignment, the dataset is organized into training, validation, and test splits using separate CSV files. This separation is maintained consistently across all experiments to ensure unbiased model evaluation and fair performance comparison.

Only a small subset of the original UCF-101 dataset is used in order to allow controlled experimentation, faster feature extraction, and systematic comparison between classical machine learning approaches and deep learning-based models.

Important: 
Customized dataset will be automatically downloaded by cloning the following GitHub repository:
https://github.com/vineet232/bits

To clone the repository run the following command:
git clone https://github.com/vineet232/bits.git