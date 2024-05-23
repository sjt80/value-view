## Introduction:

This project receives an image (jpg, png or webp) of a vinyl album record jacket to identify the album and return the suggested sale price of the item.  It does so by using a CNN for image similarity coupled with Regression techniques for optimization. 

## Requirements:

standard libraries are used with the exception of PIL, Tensorflow and Tensorflow.Keras (will add version for final submission)

The model is trained on data scraped from ebay for 5 different albums and the jupyter notebooks are split into 2 main sections:

## Step 1: EDA & Image Download:
[Notebook Link](https://github.com/sjt80/value-view/edit/main/step1-EDA+download.ipynb) 
[Google Colab Link](https://drive.google.com/file/d/10DJIZaO_zgaHYTefsnERMRjqnUUUN4-z/view?usp=sharing) 

This notebook receives a json file from our scraping; currently scraping ebay data with bright data IDE for SOLD items via a search, then we extract the product detail pages from this and rerun each pdp page to get item details. 
Running scraping on SOLD items gives us much more accurate target than a listing price however is harder as ebay protects this data more from scraping so we must use a proxy. This notebook does some initial EDA then crawls the 
url for each product image and downloads them to a local directory with use of a random residential proxy connection (so not all requests come from 1 IP that would get blacklisted by ebay). Once the images are donloaded a 
new csv file is created with an index of successfully downloaded images to run in the next notebook "pricing-item-with-image-similarity.ipynb". This is necessary as with the residential proxy some images time out due to slow 
connections so we have some images not downloaded (I need to update script to retry for failed items but have run out of time).

The current notebook can be run to get your own images from your own scraping json file however for easier demonstration purposes we have supplied some images in the repository in scraping/files/Archive.zip

## Step 2:  Value-Pred 
[Notebook Link](https://github.com/sjt80/value-view/edit/main/step2-value-pred_tf2.ipynb) 
[Google Colab Link]((https://drive.google.com/file/d/10xfO9Rx4IwqJII0-Qi2Q9FdWi3WWk-85/view?usp=sharing)) 

This is where the magic happens! This notebook predicts the price of an item based on input image with the following steps:

1.) The csv we created in the EDA data extraction notebook is imported as a csv.

2.) We set up our pretrained tensorflow CNN model and define a function to extract image embeddings with the model using predict(). Many models use the embeddings to tag or annotate images, we strip this part and just use the feature vectors / embedding extraction.

3.) The image embeddings are added back to the dataframe for the corresponding image (that in turn also represents a product/item).

4.) We extract the image embedding from a new source image or "original image".

5.) We calculate the cosine distance from each of the embeddings in our dataframe as a distance from our original image embedding and then output a list of most similar images in asccending order (lowest cosine distance first).

6.) We run the model as a batch and output embeddings to a 2D space to visualize groupings and model performance.

7.) We then implement Triplet loss function to attempt to group similar classes closer together with our model providing higher accuracy and analyse results. Also tried to implement ViT with lesser results without training model.

## Conclusion: 


### Pretrained model:

Our pre trained model already yielded very good results that yield accurate image similarity results given the variability of images. We also tried a ViT model that did not perform as well without transfer learning. Inference takes as little as 10ms so can be used in production on a small dataset.

### Fine tuned model:

Unfortunately we did not have the compute power and the kernel kept crashing until we reduced the size of the batches. We also have very limited data so the recomended steps in FaceNet to retrain the model was not ideal for our circumstances. Next steps would be to enlarge the dataset and use augmentation to train the model with a GPU. This should yield the even more accurate results we are looking for.

Budget and time permitting it would also be usefull to tune hyper parameters once we get a basic trained model but also try Vision Transformers that were tried for this but performed poorly without training.
Suggested Next steps:

Additional webscraping and annotation should be performed to train a tuned model for extracting features. Rather than using a dataframe to store our embeddings it is also recomended to use a vector database like Pinecone, Vertex (google) or other no sql systems like Elastic Search or cCsmos to store our embeddings while maintaining performance when we have large numbers.

The model can also be deployed as an app service with FastAPI or similar to receive a candidate image and return json details with price, similar images etc offering a viable consumer app to lookup the price of an item with a phone camera image!
