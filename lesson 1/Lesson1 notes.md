# Lesson 1 

## installation and basic knowledge:
!pip install -Uqq fastai to always run the latest version of the fastapi packages 
fastapi packages always start with fast____(eg fastdownload)
to obtain new models refer to "timm.fast.ai"

## Main training block:
dls = DataBlock(
blocks = (ImageBlock, CategoryBlock)
	
*here the imagesblock takes the input in this case images, and categoryblock are outputs placed in a category*
	
get_items = get_image_files
	
*gets the image files and stores int the get_items variable*
	
Splitter = RandomSplitter(valid_pct = 0.2, seed = 42)
	
*splitter splits the data into training and validation here 0.2 indicates 20% is used for validation to check whether our model can actually 	differentiate*
	
get_y = parent_label
	
*get the name of the folder the images were in*
	
item_tfms = [resize(192, method = 'squish')]
	
*used to resize the images to better fit the size the train the model and used squish instead of crop to squish images to size and not lose 	information whereas crop might lose some information*

).dataloader(path, bs=32)

## Train our Model:
learn = vision_learner(dls, resnet18, metrics = error_rate)

*this is the actual learning block of the code where the model learns given inputs are the dataloader block then the vision model used to learn and metrics*l

learn.fine_tune(3)

*fine tune is a fast ai method which automatically fine tunes a pretrained model the number in the bracket indicates the number of epochs and the number of epochs indicate the number of times the model learns to reduce errors*	
