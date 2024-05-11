Introduction:

This project receives an image (jpg, png or webp) of a vinyl album record jacket to identify the album and return the suggested sale price of the item.  It does so by using a CNN for image similarity coupled with Regression techniques. 

Requirements:

standard libraries are used with the exception of PIL, Tensorflow and Tensorflow.Keras (will add version for final submission)

The model is trained on data scraped from ebay for 5 different albums and the jupyter notebooks are split into 2 main sections:

Step 1: EDA & Image Download:

This notebook receives a json file from our scraping; currently scraping ebay data with bright data IDE for SOLD items via a search, then we extract the product detail pages from this and rerun each pdp page to get item details. 
Running scraping on SOLD items gives us much more accurate target than a listing price however is harder as ebay protects this data more from scraping so we must use a proxy. This notebook does some initial EDA then crawls the 
url for each product image and downloads them to a local directory with use of a random residential proxy connection (so not all requests come from 1 IP that would get blacklisted by ebay). Once the images are donloaded a 
new csv file is created with an index of successfully downloaded images to run in the next notebook "pricing-item-with-image-similarity.ipynb". This is necessary as with the residential proxy some images time out due to slow 
connections so we have some images not downloaded (I need to update script to retry for failed items but have run out of time).

The current notebook can be run to get your own images from your own scraping json file however for easier demonstration purposes we have supplied some images in the repository in scraping/files/Archive.zip

Step 2:  Vision+value-pred 

This is where the magic happens! This notebook predicts the price of an item based on input image with the following steps:

1.) The csv we created in the EDA data extraction notebook is imported as a csv.

2.) We set up our tensorflow CNN model and define a function to extract image embeddings with the model using predict(). Many models use the embeddings to tag or annotate images, we strip this part and just use the 
feature vectors / embedding extraction. This is essentially our training data or data to compare a new image to identify what it is.

3.) We add the image embeddings back to the dataframe for the corresponding image (that in turn also represents a product/item).

4.) We extract the image embedding from a new original image

5.) We calculate the cosine distance from the embeddings of the images in our dataframe to our original image embedding and then output a list of most similar images in descending order (with the lowest cosine distance first).

6.) We run multiple linear regression on the corresponding dataframe data and run a gridsearch to optimize the model to accurately predict the price of our item.
