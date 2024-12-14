# Object_Detection_Caltect-101
The goal is to build an object detection model using a ResNet and VGGNet architecture to classify and detect objects in images from the Caltech-101 dataset. This dataset consists of images belonging to 101 categories, including animals, vehicles, and everyday objects.

# About Dataset: Caltech - 101
The Caltech-101 dataset is a widely used dataset for object recognition tasks, containing around 9,000 images from 101 object categories. 
The categories were chosen to reflect a variety of real-world objects, and the images themselves were carefully selected and annotated to provide a challenging benchmark for object recognition algorithms.

Key Features:
The Caltech-101 dataset comprises around 9,000 color images divided into 101 categories.
The categories encompass a wide variety of objects, including animals, vehicles, household items, and people.
The number of images per category varies, with about 40 to 800 images in each category.
Images are of variable sizes, with most images being medium resolution.
Caltech-101 is widely used for training and testing in the field of machine learning, particularly for object recognition tasks.

# Procedure to execute
Run each cell in the ipynb file, upload the kaggle.json file to download the dataset and execute the rest of the cells sequentially. 

# Overall Explanation:
Step 1: Load and Split the Dataset
Load the Caltech-101 dataset from kaggle using ImageFolder and apply transformations like resizing, normalization, and data augmentation.
Load the kaggle.json file provided to load the dataset.
Split the dataset into training, validation, and test sets. (0.70 - training, 0.15 - validation, 0.15 - testing)
Step 2: Create Data Loaders
Create data loaders for the training, validation, and test datasets. 

Step 3: Set Up the Model
Load a pre-trained ResNet and VGGNet model from PyTorch.
Modify the last fully connected layer to match the number of classes (101 classes) in the Caltech-101 dataset (output should match the number of object categories).

Step 4: Define Loss Function and Optimizer
Set up the loss function using CrossEntropyLoss for multi-class classification.
Define an Adam optimizer to adjust the model’s weights based on the gradients.
Add a learning rate (0.001) scheduler that reduces the learning rate during training to improve performance.

Step 5: Train the Model
Train the model over several epochs. 
For each epoch:
Pass batches of images through the model.
Calculate the loss between predicted labels and true labels.
Update the model’s weights using backpropagation.
Track the loss for each epoch to monitor the learning process.
Use the learning rate scheduler to adjust the learning rate after every few epochs.

Step 6: Validate the Model
After each epoch, evaluate the model on the validation set to check how well it's learning. Track the accuracy and loss for validation.

Step 7: Plot the Learning Curve and Confusion matrix
After training, plot a graph of the training loss vs. epochs to visualize how the loss decreases over time.
Plot the confusion matrix to evaluate the model’s performance.

Step 8: Test the Model
After training is complete, evaluate the model on the test set to check its final performance.

Step 9: Save the Model
Once we are done with training the model, save it to disk so that you can load it later without retraining.
