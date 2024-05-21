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

Step 2:  Value-Pred 

This is where the magic happens! This notebook predicts the price of an item based on input image with the following steps:

1.) The csv we created in the EDA data extraction notebook is imported as a csv.

2.) We set up our pretrained tensorflow CNN model and define a function to extract image embeddings with the model using predict(). Many models use the embeddings to tag or annotate images, we strip this part and just use the feature vectors / embedding extraction.

3.) The image embeddings are added back to the dataframe for the corresponding image (that in turn also represents a product/item).

4.) We extract the image embedding from a new source image or "original image".

5.) We calculate the cosine distance from each of the embeddings in our dataframe as a distance from our original image embedding and then output a list of most similar images in asccending order (lowest cosine distance first).

6.) We run the model as a batch and output embeddings to a 2D space to visualize groupings and model performance.

7.) We then implement Triplet loss function to attempt to group similar classes closer together with our model providing higher accuracy and analyse results. Also tried to implement ViT with lesser results without training model.

8.) Conclusion: see below for outcomes and next steps with recomendations
