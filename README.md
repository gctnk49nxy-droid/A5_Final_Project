# A5_Final_Project

DTI5125 Group Assignment 5 Final Project
Names: Dawson Oliver (300515143), Yee Mon Thant (300332907), and Praveen Pullaperuma (8908907)
Class: DTI5125: Data Science Applications
Professor: Dr. Arya Rahgozar
TA: TA Pouria Mortezaagha

## Introduction

There are 5 main python files associated with this project:

1. 0_Dataset_Creation_Text_Processing.ipynb: This notebook creates the final dataset the group assignment 5. This dataset is provided as part of the project, you do not need to run the code, but it is available for your interest. The code extracts a portion of the entire dataset used for this project which contains Amazon Reviews for products in the electronics category. This code also completes basic text preprocessing such as lemmatization, stemming, stop word removal, removal of special characters, and removal of punctuation. There were three different cleaned text columns: cleamed_lemma_text, cleaned_text, and cleaned_for_aspect that were slightly different to accomodate for different analyses.

2. 1_Feature_Engineering.ipynb: This notebook implements the full feature engineering pipeline for the recommender system, considsting of aspect extraction and aspect sentiment analysis, aspect normalization/category mapping, product-level aspect profiles, text-based feature vectorization, knowledge graph construction and product node embeddings.

3. 2_Clustering.ipynb: This notebook implements the clustering component for the recommender system

4. 3_Classification.ipynb: This notebook implements the the classification of predicting the rating group at a product level.

5. 4_Recommendation.ipynb: This notebook implements the recommendation engine that produces the product-level recommendations. It does this through a variety of methods, including cosine similarity, clustering results, and aspect information.

## Important Notes
- For this product, the python files `0_Dataset_Creation_Text_Processing.ipynb`, `1_Feature_Engineering.ipynb`, `3_Classification.ipynb`, and `4_Recommendation.ipynb` all use one python environment: *a5_env_1* whereas `2_Clustering.ipynb` uses another python environment: *a5_env_2*. The reason for two environemnts is due to Python package dependent requirements being different between the datamapplot package and the Word2Vec/Node2Vec package.
- If you manually import umap you must use the code: `pip install umap-learn`
- `1_Feature_Engineering.ipynb` takes a very long time to run due to the aspect extraction and sentiment analysis. We have included the dataset `aspects_raw.parquet` to avoid running the Batch Aspect & Sentiment Extraction and Save Raw Aspects & Summary Statistics sections out of the sake of time.
- Please use python version 3.11 to avoid errors

## Base Environment Setup

This section will describe how to create the two base environments used for this project. Python package installation will be described in the following section.

Step 1: Start with an empty conda environment to avoid dependency conflicts:

In the python terminal go:
`conda create --name a5_env_1 python=3.11 ipykernel`
If prompted, select y and enter to proceed

activate environment in python terminal go:
`conda activate a5_env_1`

Register environment as a jupyter kernel
`python -m ipykernel install --user --name a5_env_1 --display-name "a5_env_1"`

Close the terminal window

*Completed with the first base environment, repeat steps for the second base environment*

Step 2: Start with an empty conda environment to avoid dependency conflicts:

In the python terminal go:
`conda create --name a5_env_2 python=3.11 ipykernel`
If prompted, select y and enter to proceed

activate environment in python terminal go:
`conda activate a5_env_2`

Register environment as a jupyter kernel
`python -m ipykernel install --user --name a5_env_2 --display-name "a5_env_2"`

Close the Terminal window

Step 2: Import ipynb notebooks into your environment and select the kernel to be "a5_env_1" for files  `0_Dataset_Creation_Text_Processing.ipynb`, `1_Feature_Engineering.ipynb`, `3_Classification.ipynb`, and `4_Recommendation.ipynb` and "a5_env_2" for the file `2_Clustering.ipynb` 

Step 3: Install required libraries. There are three methods that will be detailed to complete this step if one method fails, please try another. 

Method 1: At the top of the associated ipynb file, there is a code chunk that when run, will install and import all required libraries for that environemnt. To complete this run this code chunk. If there are errors, either manually install problematic packages or default to another of the three methods provided. 

Method 2: 

Method 3: 



