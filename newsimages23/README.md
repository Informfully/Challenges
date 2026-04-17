# NewsImages at MediaEval 2023

Repository for the team DDIS group submission to the MediaEval 2023 NewsImages Challenge.
We share both our NewsImages 2023 [challenge runs](https://github.com/Informfully/Challenges/tree/main/newsimages23/runs) as well as our [user study images](https://github.com/Informfully/Challenges/tree/main/newsimages23/images) and their [evaluation](https://github.com/Informfully/Challenges/tree/main/newsimages23/evaluation) for the Quest for Insight paper.

Selecting appropriate images for news articles remains a manual and time-consuming task for journalists.
This project aims to expedite the process of identifying relevant images based on news article headlines or content.

Our work is motivated by the challenges posed in the MediaEval 2023 competition, specifically the [NewsImages: Connecting Text and Images task](https://multimediaeval.github.io/editions/2023/tasks/newsimages).
This competition focuses on understanding the relationship between textual and visual content in news articles. Participants are tasked with developing models to describe and predict the connections between images and textual elements (both headlines and snippets) in news articles.

## Dataset

The dataset for this study was sourced from the MediaEval 2023 NewsImages Task dataset.
This dataset comprises a rich collection of news articles from various online news sources, each paired with its original accompanying image.
Each article in the dataset includes the title, tags, and the original image, providing a comprehensive set of multimodal data.
The dataset is divided into three subsets: GDELT1, GDELT2 and RT.

During the training phase, participants were given datasets with news headlines and their corresponding images to train their models.
While in testing phase, participants received a new set of news headlines without images and a pool of shuffled images.
The task was to match the headlines with the correct images from the pool.

## Evaluation Metrics

The quality of the article-image mapping is assessed using the Hits@K metric, which evaluates the effectiveness of our model by determining if the correct image for an article appears within the top k ranked results.
If the correct image is found within the top k predicted images, it is considered a "hit".
The final Hits@k score is the average number of hits across all articles.

## Approaches

After extensive research, we adopted the CLIP model [developed by OpenAI](https://openai.com/index/clip), as our base methodology.
The CLIP model, trained on 400 million text-image pairs, has demonstrated exceptional performance in text-image matching tasks and covers a diverse range of domains, making it highly generalizable.
For this project, we used the **clip-vit-base-patch32** version of the CLIP model.

Given the relatively small size of our dataset compared to the CLIP model's training dataset, and our experiments showing that fine-tuning the CLIP model does not surpass its original performance, we focused on tuning the dataset by rephrasing the data.
The motivation is to enhance the compatibility of our data with the CLIP model, aiming to achieve better performance in matching relevant images based on rephrased headlines.

We employed seven different methods:

1. **Raw Title**: The original title of the news article.
2. **Pre-processed Title**: The title of the news article after data preprocessing.
3. **Tags**: Additional information related to the news article.
4. **Raw Title + Tags**: A combination of the pre-processed title and tags.
5. **TextRank**: Extraction of keywords from the article title using the TextRank algorithm.
6. **Named Entity Recognition (NER) + TextRank**: Extraction of keywords/entities using NER, sorted based on the TextRank algorithm, with varying numbers of keywords (6, 7, 8, 9, 10).
7. **T5**: Rewriting the article title using a pre-trained T5 model.

## Files

### Training

- `{subdataset}_Precision@K`: The main Python script that performs image retrieval.
- `{subdataset}.zip`: File containing the pool of images.
- `{subdataset}.txt`: File containing the dataset with news headlines, tags, image names, and additional metadata.
- `{subdataset}_Train_Result.csv`: File containing the intermediate results (rephrased titles from different approaches) to handle potential interruptions.

### Testing

- `{subdataset}_Submission`: The main Python script that performs image retrieval.
- `{subdataset}.zip`: File containing the pool of images.
- `{subdataset}-Test-Text.txt`: File containing the news headlines, tags and additional text-related metadata.
- `{subdataset}-Test-Img.txt`: File containing the image metadata.
- `{subdataset}_Result.csv`: File containing the intermediate results (rephrased titles from different approaches) to handle potential interruptions. 
- `Submission Files/{subdataset}_{data-tuning approach}.txt`: Contains the top 100 relevant images, sorted from most to least relevant, based on various data-tuning approaches.

## Usage

Our code is written in Python and can be run in Jupyter Notebook. Follow these steps to set up and execute the image retrieval process:

1. Prepare the Dataset: Ensure that you have the dataset ready and placed it in the appropriate directory. Ensure you have extract the contents of the .zip file.
The images should be located directly within subfolders of the parent directory.
For example, images should reside in `parentfolder/subfolder`, and **NOT** in `parentfolder/subfolder/subfolder`.

2. Run the Script: Execute the cells in the notebook sequentially to perform image retrieval. Modify the paths and parameters as needed based on your dataset.

Parameters and Configuration:

- **file_dir**: Path to the directory containing the .txt and .zip files.
- **image_folder_dir**: Path to the folder containing the image pool.
- **saveCheckpoint** (Optional): Save intermediate results (rephrased titles from different approaches) to handle potential interruptions.
- **loadLastCheckpoint** (Optional): Restore intermediate results from a previous save.

## Code Overview

### Rephrasing the Article Headline Based on the 7 Approaches

1. **Raw Title**
2. **Pre-processed Title**
3. **Raw Title + Tags**
4. **TextRank**
5. **Named Entity Recognition (NER) + TextRank**
6. **T5**

### Convert Headline into Embeddings Using CLIP

Leverage the text embedding from CLIP, converting the rephrased headline to embeddings.

### Convert Image into Embeddings Using CLIP

Leverage the image embedding from CLIP, converting all images from the pool to embeddings.

### Calculate and Sort Based on Cosine Similarity Between Text Embedding and Image Embedding

Calculate cosine similarity between text embedding and image embedding, and sort the images from highest similarity to lowest similarity.

### Calculate hit@K score (1,5,10,20,50,100)

To evaluate the performance of our model, we calculate the Hits@K score for various values of K (1, 5, 10, 20, 50, 100). The Hits@K score is determined as follows:

- Predicted Output: The model generates a sorted list of images based on their similarity scores to the given headline.
- Original Output: The correct image associated with the headline.

For each headline:

- Check if the correct image (original output) is within the top K images in the predicted output.
- If the correct image is found within the top K results, it is considered a hit, scored as 1.
- If the correct image is not within the top K results, it is considered a miss, scored as 0.

### Calculate average the hit@K score

Average the Hits@K scores across all images by summing the Hits@K scores for all images and divide the sum by the total number of images. The average Hits@K score is ranged between 0 and 1.

## Output

The script produces the following outputs: Average Hit@K score for different approaches.

## Acknowledgement

- MediaEval 2023 for providing the dataset and competition task.
- OpenAI for developing the CLIP model.

If you are using any data from this repository, we ask you to cite our Quest for Insight Paper and Working Notes Paper.

- [An Empirical Exploration of Perceived Similarity between News Article Texts and Images](https://ceur-ws.org/Vol-3658/paper8.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2023 Workshop, 2024.

  ```tex
  @inproceedings{heitz2024empirical,
  title={An Empirical Exploration of Perceived Similarity between News Article Texts and Images},
  author={Heitz, Lucien and Rossetto, Luca and Bernstein, Abraham},
  booktitle={Working Notes Proceedings of the MediaEval 2023 Workshop},
  year={2024}
  }
  ```

- [Prompt-Based Alignment of Headlines and Images Using OpenCLIP](https://ceur-ws.org/Vol-3658/paper7.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2023 Workshop, 2024.

  ```tex
  @inproceedings{heitz2024prompt,
  title={rPompt-Based Alignment of Headlines and Images Using OpenCLIP},
  author={Heitz, Lucien and Chan, Yuin-Kwan and Li, Hongji and Zeng, Kerui and Rossetto, Luca and Bernstein, Abraham},
  booktitle={Working Notes Proceedings of the MediaEval 2023 Workshop},
  year={2024}
  }
  ```
