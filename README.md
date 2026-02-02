# NewsImages at MediaEval 2026

README file for the data overview and expected submission of the NewsImages Challenge at MediaEval 2026.
Please refer to the official [MediaEval 2026 website](https://multimediaeval.github.io/editions/2026/) for the full task description and event registration details.

> Note: This task summary only applies to regular paper submissions to the NewsImages challenge.
If you are writing a [Quest for Insight](https://multimediaeval.github.io/editions/2026/tasks/newsimages) paper, then you can skip this outline and go directly to the deadlines below.

## Task Summary

Upon successful registration, the participants are given a collection of 8,500 news articles with images (the article text is in English, from [GDELT](https://www.gdeltproject.org).
Given a randomly selected article, the goal is to build a pipeline that combines image retrieval and image generation techniques to provide a **fitting** image recommendation for a given news article text.
There will be a crowdsourced online event where all participating teams take part in rating the submitted image recommendations using a 5-point Likert scale.

The winning team is determined by the **highest average image rating** for the articles within the evaluation dataset.
This evaluation dataset will be shared with all **registered groups**, together with the information on where to submit your results (see deadlines below).

## Data Overview

The *training* dataset you receive includes the list of 8,500 article IDs from the 2025 iteration of the task.
This allows you, for example, to create image recommendations for the same images and compare your approach with [last year's results](https://github.com/Informfully/Challenges/tree/main/newsimages25/images) in a prelimiary user study.

The *test* dataset for the online evaluation event contains around 850 articles.
It may include articles from the training dataset and articles from the evaluation events of previous iterations.
Both datasets are contains a CSV with the following data on news articles:

| Attribute | Description |
| - | - |
| article_id | ID of news article. |
| article_url | Original URL of the news article. |
| article_title | Title of the news article (may include lead). |
| image_id | ID of news image (we provide a copy of the image). |
| image_url | Original URL of the news image. |

Furthermore, a folder 'newsimages' containing a copy of the original thumbnails is included.
The name of each JPG file corresponds to the 'image_id' associated with each news article.

For image retrieval, we recommend using [Yahoo-Flickr Creative Commons 100 Million (YFCC100M)](https://www.multimediacommons.org).
You are free to use any other open-source dataset, but this must be shared with us as part of your workflow submission.

## Expected Submission


Your submission can include multiple runs/approaches.
Each approach must include *precisely one* image recommendation for a given article ID.
(If there is no valid recommendation for a given article, the recommendation will receive a default rating score of 0.)

> Important: There is no restriction in terms of how many runs you can submit. However, all the runs need to be sufficiently different from one another.
*No two runs can have the same image recommendation for a given article ID.*

To submit your runs, you must provide a ZIP file [group_name].zip that is structured as follows:

[group_name] + _ + [approach_name] / [article_id] + _ + [group_name] + _ + [approach_name].png

Use the group name with which you have registered for the task.
For each submitted approach/run, please provide a **unique name** (i.e., unique among your runs).

You must hand in your workflow together with the Working Notes Papers (see deadlines below).
Please have a look at [last year's workflows](https://github.com/Informfully/Challenges/tree/main/newsimages25/workflows) to get an idea of how to organize your codebase.

### Example Group Submission

Below is the folder structure of an example ZIP file for the group 'UnstableOsmosis':

    UnstableOsmosis_Submission.zip
    |_ UnstableOsmosis_FLUX
    |  |_ 117_UnstableOsmosis_ZImage.png
    |  |_ …
    |_ UnstableOsmosis_OpenCLIP
    |  |_ 117_UnstableOsmosis_OpenCLIP.png
    |  |_ …
    |_ …

### Required Image Format

The image format must be PNG, with a target dimension of 460x260 pixels (in landscape orientation).
This applies to both generated and retrieved images.
If you generate images with tools like [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and then edit them afterwards (e.g., for cropping), ensure the workflow **remains** embedded.
We advise teams not to stretch the images.
Consider simply rescaling the images if cropping is not an option.
(And instead of using black bars to reframe the image, you can leave the background transparent with PNGs.)

### Complete Email Submission

You will need to submit your runs by the deadlines indicated below.
Do that by sending an email to the address that shared the dataset download link with you.
It must include (1) your group name, (2) a link to download your image submissions, and (3) links to the documented code of your workflow (e.g., a link to a GitHub repository with a notebook and/or a collection of scripts).
(Please note that this is something separate from the Working Notes Paper.)

### Previous Workflow Examples

We advise all teams to carefully study previous submissions.
We share the code of all [previous workflows](https://github.com/Informfully/Challenges/tree/main/newsimages25) together with their [overview papers](https://2025.multimediaeval.com).
Please follow the recommendations outlines in the [Task Overview Paper (Release February)](TBD) to ensure that you have a competitive run submission.

## Online Evaluation

Taking part in the online evaluation event is mandatory.
During the evaluation, participating teams rate the image recommendations of other teams.
To do that, they are being presented with a news headline and two image recommendations.
They then need to select which image is more fitting.
These ratings are used to calculate an overall ranking of images for each article.
The average rank of team submissions across the featured item pool then determines the overall winner of the challenge.

## Working Notes Paper

As part of the challenge submission, each team is required to write a separate **Working Notes Paper** that documents and outlines their approach.
Please refer to the [online paper template](https://drive.google.com/drive/folders/1DNhxIeACfsmg6rrdgQZ22BbRtYE8ioYI) for additional information.

We encourage open and reproducible science.
We ask each team to share their codebase and workflows.
Please use the examples in the [designated folder](https://github.com/Informfully/Challenges/tree/main/newsimages26/workflows) to structure your code and make a pull request to contribute your workflow.

Note that each team will be asked to take part in the review process of the papers on EasyChair.
Make sure you are available between the paper submission and review deadlines (see the overview below).
Furthermore, we ask each group to include and refer to the following papers in their Working Notes Paper:

<!-- * [NewsImages in MediaEval 2026 – Automated Image Recommendations with Retrieval and Generation Techniques for News Articles](TBD), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2026 Workshop, 2026.-->

* [NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles](https://2025.multimediaeval.com/paper1.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2025 Workshop, 2025.

  ```tex
  @inproceedings{heitz2025newsimages,
  title={NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles},
  author={Heitz, Lucien and Rossetto, Luca and Kille, Benjamin and Lommatzsch, Andreas and Elahi, Mehdi and Dang-Nguyen, Duc-Tien},
  booktitle={Working Notes Proceedings of the MediaEval 2025 Workshop},
  year={2025}
  }
  ```

* [An Empirical Exploration of Perceived Similarity between News Article Texts and Images](https://ceur-ws.org/Vol-3658/paper8.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2023 Workshop, 2024.

  ```tex
  @inproceedings{heitz2024empirical,
  title={An Empirical Exploration of Perceived Similarity between News Article Texts and Images},
  author={Heitz, Lucien and Rossetto, Luca and Bernstein, Abraham},
  booktitle={Working Notes Proceedings of the MediaEval 2023 Workshop},
  year={2024}
  }
  ```

## Deadline Summary (TBD)

* Registration opening and release train dataset: February 1
* Release test dataset: April 1
* Registration closes: April 24
* Runs due: May 1 (AoE, incl. workflows to reproduce your image recommendations)
* Online evaluation: May 7-14 (with Qualtrics, no user account required)
* Evaluation feedback: May 21
* Working Notes Paper submission: May 31* (eith EasyChair, user account required)
* Review deadline: June 7**
* Camera-ready deadline: June 14 (AoE)
* MediaEval workshop: June 15-16, co-located with ACM ICMR 2026 (more information on the [registration website](https://multimediaeval.github.io/editions/2026), in-person or online attendance required).

(*) We provide you with a review/feedback for your paper within one week of submission (by June 7).
Afterwards, you then have another week to prepare the camera-ready revision by June 14 (exact deadlines will be communicated by the MediaEval organizers).
Please note that your paper should include a results section that is based on your performance in the online evaluation.
The necessary information for this part will be forwarded to you after the evaluation event has concluded on May 21.
The Working Notes Paper **must** describe the workflows for your submissions.
It **may** include complementary and/or alternative approaches that you tested.
We also encourage all teams to write a separate "Quest for Insight" paper if there are interesting findings you would like to share and discuss with (for more information, see "Quest for Insight" in our challenge overview: <https://multimediaeval.github.io/editions/2026/tasks/newsimages>).
