# MediaEval2025

This repo targets the NewsImages: Retrieval and generative AI for news thumbnails task of the [MediaEval 2025](https://multimediaeval.github.io/editions/2025/tasks/newsimages/). This repo is for the generation task, news images are generated using a small and large dataset.



## Small task

1. Use the notebook **MediaEval2025\_GEN\_VIVID\_SMALL.ipynb** and edit it to provide location of the **subset.csv** file and run the file. Current workflow is set for Google Colab environment where Google Drive is used for data storage

2. Use the notebook **MediaEval2025\_GEN\_renaming\_images.ipynb** to rename the stored images. The images should be made available to the source directory (path may be edited accordingly)

3. During this task a CSV file will be generated **image\_seed\_mapping\_small.csv**, this may be used to reproduce the generated images using the notebook



## Large task

1. First step is to modify prompts using the published news articles, for this task use the notebook **MediaEval2025\_prompt\_generation\_PromptForge\_LARGE.ipynb** to generate the **newsarticles\_with\_prompts.csv**. This file may later modified to concatenate the *article\_title*, *article\_tags* and *positive\_prompts* in a separate column. This step is necessary especially for the articles which have no URL and data extraction was unsuccessful. A final file is already stored as **data/newsarticles\_with\_prompts.xlsx**

2. Use the notebook **MediaEval2025_GEN_PromptForge_LARGE.ipynb** and edit it to provide location of the **newsarticles\_with\_prompts.xlsx** file and run the file. Current workflow is set for Google Colab environment where the Google Drive is used for data storage. You may generate images in batches of articles and you may generate batches of images per article.

3. Use the notebook **MediaEval2025\_GEN\_renaming\_images.ipynb** to rename the stored images. The images should be made available to the source directory (path may be edited accordingly)

4. During this task a CSV file will be generated **image\_seed\_mapping\_large.csv**, this may be used to reproduce the generated images using the notebook
