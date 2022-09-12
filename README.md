# BLA: a benchmark for Basic Language Abilities

<!-- ABOUT THE PROJECT -->
## About The Project
BLA: a benchmark for Basic Language Abilities is proposed to explore the linguistic knowledge of pretrained vision-and-language models. The benchmark evaluates the models on their understanding of active-passive voices, coordination and sentences embedding a relative clause, three linguistic phenomena that are commonly used in English. And it proposes a framework to tease apart, as much as possible, linguistic vs. reasoning abilities of models and design the evaluation tasks more similar to the tests for humans. 



<!-- Dataset -->
## About the Dataset
Please find the data in the "data" folder. The dataset is in json format and contains the following relevant fields:
"image_id": the image id of the caption group
"True1": caption 1 that correctly describes the image
"True2": caption 2 that correctly describes the image
"False1": distractor 1 that is incorrect in relation the the image
"False2": distractor 2 that is incorrect in relation the the image

Each json file contains one task of the benchmark. The images are in the folder "images". Check the "dataset_demo_*" files for statistics and examples on how to use each dataset.

