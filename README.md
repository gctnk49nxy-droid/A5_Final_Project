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

Method 1: At the top of the associated ipynb file, there is a code chunk that when run, will install and import all required libraries for that environemnt. *Please ensure that you have the correct environment for that file*. To complete this run this code chunk. If there are errors, either manually install problematic packages or default to another of the three methods provided. Please ensure to restart the kernel after the installation to prevent any additional errors.

Method 2: You can use the requirements files included at the bottom of this notebook to create the libraries:
For a5_env_1:

Step 1: Go to the terminal and type `conda activate a5_env_1`

Step 2: Ensure that the requirements_a5_env_1.txt file is within the same folder as your python notebooks and enter `pip install -U -r requirements_a5_env_1`

Similarly for a5_env_2:

Step 1: Go to the terminal and type `conda activate a5_env_2`

Step 2: Ensure that the requirements_a5_env_2.txt file is within the same folder as your python notebooks and enter `pip install -U -r requirements_a5_env_2`

Method 3: Manually for each library in the terminal enter `pip install library_name` for each library used for the associated environment, this will need to be completed for both a5_env_1 and a5_env_2. Please look at the associated requirements file below for the associated libraries.

### requirements_a5_env_1.txt:
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
appnope @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_46iquy4y0r/croot/appnope_1750774774351/work
asttokens @ file:///Users/ec2-user/croot/asttokens_1773135889661/work
attrs==26.1.0
autocuda==0.16
blis==1.3.3
bokeh==3.9.0
boostaug==2.3.5
catalogue==2.0.10
certifi==2026.2.25
charset-normalizer==3.4.7
click==8.3.2
cloudpathlib==0.23.0
cloudpickle==3.1.2
colorama==0.4.6
colorcet==3.1.0
colorspacious==1.1.2
comm @ file:///opt/miniconda3/conda-bld/comm_1763119153703/work
confection==1.3.3
contourpy==1.3.3
cycler==0.12.1
cymem==2.0.13
dask==2026.3.0
datamapplot==0.7.1
datashader==0.19.0
debugpy @ file:///opt/miniconda3/conda-bld/debugpy_1762421636036/work
decorator @ file:///opt/miniconda3/conda-bld/decorator_1757341235959/work
distributed==2026.3.0
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
et_xmlfile==2.0.0
executing @ file:///opt/miniconda3/conda-bld/executing_1757061235776/work
fastjsonschema==2.21.2
filelock==3.25.2
findfile==2.1.2
fonttools==4.62.1
fsspec==2026.3.0
gensim==4.4.0
gitdb==4.0.12
GitPython==3.1.46
h11==0.16.0
hdbscan==0.8.42
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.11
ImageIO==2.37.3
importlib_metadata==9.0.0
importlib_resources==6.5.2
ipykernel @ file:///Users/ec2-user/croot/ipykernel_1772612352018/work
ipython @ file:///opt/miniconda3/conda-bld/ipython_1762789298678/work
ipython_pygments_lexers @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_25fteymumy/croot/ipython_pygments_lexers_1744753256449/work
ipywidgets==8.1.8
jedi @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_38ctoinnl0/croot/jedi_1733987402850/work
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_client @ file:///Users/ec2-user/croot/jupyter_client_1768399009291/work
jupyter_core @ file:///opt/miniconda3/conda-bld/jupyter_core_1764588284546/work
jupyterlab_widgets==3.0.16
kiwisolver==1.5.0
lazy-loader==0.5
llvmlite==0.47.0
locket==1.0.0
lz4==4.4.5
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
matplotlib-inline @ file:///opt/miniconda3/conda-bld/matplotlib-inline_1762779171588/work
mdurl==0.1.2
metric_visualizer==0.9.17
mpmath==1.3.0
msgpack==1.1.2
multipledispatch==1.0.0
murmurhash==1.0.15
narwhals==2.18.1
natsort==8.4.0
nbformat==5.10.4
nest-asyncio @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_310vb5e2a0/croot/nest-asyncio_1708532678212/work
networkx==3.6.1
nltk==3.9.4
node2vec==0.5.0
numba==0.65.0
numpy==1.26.4
openpyxl==3.1.5
packaging==26.0
pandas==2.3.3
param==2.3.3
parso @ file:///opt/miniconda3/conda-bld/parso_1762781703292/work
partd==1.4.2
pexpect @ file:///opt/miniconda3/conda-bld/pexpect_1762535935939/work
pillow==12.2.0
platformdirs @ file:///Users/ec2-user/croot/platformdirs_1773652975053/work
plotly==6.6.0
preshed==3.0.13
prompt_toolkit @ file:///opt/miniconda3/conda-bld/prompt-toolkit_1761744881146/work
protobuf==3.20.3
psutil @ file:///opt/miniconda3/conda-bld/psutil_1761896323688/work
ptyprocess @ file:///opt/miniconda3/conda-bld/ptyprocess_1762424170819/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl#sha256=3470be7f810474c8a2ecfcd6e02acc6aea8483ab595417fa4e336362a349933e
pure_eval @ file:///opt/miniconda3/conda-bld/pure_eval_1757067067189/work
pyabsa==2.4.2
pyarrow==23.0.1
pyct==0.6.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments @ file:///Users/ec2-user/croot/pygments_1775127021103/work
pylabeladjust==0.1.13
pynndescent==0.6.0
pyparsing==3.3.2
Pyqtree==1.0.0
python-dateutil @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_66ud1l42_h/croot/python-dateutil_1716495741162/work
pytorch-warmup==0.2.0
pytz==2026.1.post1
PyYAML==6.0.3
pyzmq @ file:///opt/miniconda3/conda-bld/pyzmq_1762375616040/work
rcssmin==1.2.2
referencing==0.37.0
regex==2026.4.4
requests==2.33.1
rich==14.3.3
rjsmin==1.2.5
rpds-py==0.30.0
safetensors==0.7.0
scikit-image==0.26.0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
sentencepiece==0.2.1
seqeval==1.2.2
shellingham==1.5.4
six @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_03myqm7p6o/croot/six_1744271511946/work
smart_open==7.5.1
smmap==5.0.3
sortedcontainers==2.4.0
spacy==3.8.14
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.3
stack_data @ file:///opt/miniconda3/conda-bld/stack_data_1757067039886/work
sympy==1.14.0
tabulate==0.10.0
tblib==3.2.2
termcolor==3.3.0
thinc==8.3.13
threadpoolctl==3.6.0
tifffile==2026.3.3
tikzplotlib==0.10.1
tokenizers==0.13.3
toolz==1.1.0
torch==2.11.0
tornado @ file:///Users/ec2-user/croot/tornado_1774303634539/work
tqdm==4.67.3
traitlets @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_500m2_1wyk/croot/traitlets_1718227071952/work
transformers==4.29.0
typer==0.24.1
typing-inspection==0.4.2
typing_extensions @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_dffmcwqo6o/croot/typing_extensions_1756281471323/work
tzdata==2026.1
umap-learn==0.5.11
update-checker==0.18.0
urllib3==2.6.3
wasabi==1.1.3
wcwidth @ file:///Users/ec2-user/croot/wcwidth_1767780856263/work
weasel==1.0.0
webcolors==25.10.0
widgetsnbextension==4.0.15
wrapt==2.1.2
xarray==2026.2.0
xlsxwriter==3.2.9
xyzservices==2026.3.0
zict==3.0.0
zipp==3.23.0
### requirements_a5_env_2.txt:
appnope @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_46iquy4y0r/croot/appnope_1750774774351/work
asttokens @ file:///Users/ec2-user/croot/asttokens_1773135889661/work
bokeh==3.9.0
certifi==2026.2.25
charset-normalizer==3.4.7
click==8.3.2
cloudpickle==3.1.2
colorcet==3.1.0
colorspacious==1.1.2
comm @ file:///opt/miniconda3/conda-bld/comm_1763119153703/work
contourpy==1.3.3
cycler==0.12.1
dask==2026.3.0
datamapplot==0.7.1
datashader==0.19.0
debugpy @ file:///opt/miniconda3/conda-bld/debugpy_1762421636036/work
decorator @ file:///opt/miniconda3/conda-bld/decorator_1757341235959/work
distributed==2026.3.0
executing @ file:///opt/miniconda3/conda-bld/executing_1757061235776/work
fonttools==4.62.1
fsspec==2026.3.0
hdbscan==0.8.42
idna==3.11
ImageIO==2.37.3
importlib_metadata==9.0.0
importlib_resources==6.5.2
ipykernel @ file:///Users/ec2-user/croot/ipykernel_1772612352018/work
ipython @ file:///opt/miniconda3/conda-bld/ipython_1762789298678/work
ipython_pygments_lexers @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_25fteymumy/croot/ipython_pygments_lexers_1744753256449/work
jedi @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_38ctoinnl0/croot/jedi_1733987402850/work
Jinja2==3.1.6
joblib==1.5.3
jupyter_client @ file:///Users/ec2-user/croot/jupyter_client_1768399009291/work
jupyter_core @ file:///opt/miniconda3/conda-bld/jupyter_core_1764588284546/work
kiwisolver==1.5.0
lazy-loader==0.5
llvmlite==0.47.0
locket==1.0.0
lz4==4.4.5
MarkupSafe==3.0.3
matplotlib==3.10.8
matplotlib-inline @ file:///opt/miniconda3/conda-bld/matplotlib-inline_1762779171588/work
msgpack==1.1.2
multipledispatch==1.0.0
narwhals==2.19.0
nest-asyncio @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_310vb5e2a0/croot/nest-asyncio_1708532678212/work
networkx==3.6.1
numba==0.65.0
numpy==2.4.4
packaging==26.0
pandas==2.3.3
param==2.3.3
parso @ file:///opt/miniconda3/conda-bld/parso_1762781703292/work
partd==1.4.2
pexpect @ file:///opt/miniconda3/conda-bld/pexpect_1762535935939/work
pillow==12.2.0
platformdirs @ file:///Users/ec2-user/croot/platformdirs_1773652975053/work
plotly==6.6.0
prompt_toolkit @ file:///opt/miniconda3/conda-bld/prompt-toolkit_1761744881146/work
psutil @ file:///opt/miniconda3/conda-bld/psutil_1761896323688/work
ptyprocess @ file:///opt/miniconda3/conda-bld/ptyprocess_1762424170819/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl#sha256=3470be7f810474c8a2ecfcd6e02acc6aea8483ab595417fa4e336362a349933e
pure_eval @ file:///opt/miniconda3/conda-bld/pure_eval_1757067067189/work
pyarrow==23.0.1
pyct==0.6.0
Pygments @ file:///Users/ec2-user/croot/pygments_1775127021103/work
pylabeladjust==0.1.13
pynndescent==0.6.0
pyparsing==3.3.2
Pyqtree==1.0.0
python-dateutil @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_66ud1l42_h/croot/python-dateutil_1716495741162/work
pytz==2026.1.post1
PyYAML==6.0.3
pyzmq @ file:///opt/miniconda3/conda-bld/pyzmq_1762375616040/work
rcssmin==1.2.2
requests==2.33.1
rjsmin==1.2.5
scikit-image==0.26.0
scikit-learn==1.8.0
scipy==1.17.1
six @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_03myqm7p6o/croot/six_1744271511946/work
sortedcontainers==2.4.0
stack_data @ file:///opt/miniconda3/conda-bld/stack_data_1757067039886/work
tblib==3.2.2
threadpoolctl==3.6.0
tifffile==2026.3.3
toolz==1.1.0
tornado @ file:///Users/ec2-user/croot/tornado_1774303634539/work
tqdm==4.67.3
traitlets @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_500m2_1wyk/croot/traitlets_1718227071952/work
typing_extensions @ file:///private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_dffmcwqo6o/croot/typing_extensions_1756281471323/work
tzdata==2026.1
umap-learn==0.5.11
urllib3==2.6.3
wcwidth @ file:///Users/ec2-user/croot/wcwidth_1767780856263/work
xarray==2026.2.0
xyzservices==2026.3.0
zict==3.0.0
zipp==3.23.0




