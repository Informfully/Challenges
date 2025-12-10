# NewsImages at MediaEval 2026

README file for the data overview and expected submission of the NewsImages Challenge at MediaEval 2026.
Please see the official [MediaEval 2026 website](TBD) for the full task description and the event registration.

## Task Summary

Upon successful registration, the participants are given a collection of 8,500 news articles with images (the article text is in English, from [GDELT](https://www.gdeltproject.org).
Given a randomly selected article, the goal is to build a pipeline that combined image retrieval and image generation techniques to provide a **fitting** image recommendation for a given news article text.
There will be a crowdsourced online event where all participating teams take part in rating the submitted image recommendations using a 5-point Likert scale.

The winning team is determined by the **highest average image rating** for the articles within the evaluation dataset.
This evaluation dataset will be shared with all **registered groups**, together with the information on where to submit your results (see deadlines below).

## Data Overview

The challenge data contains a CSV with the following data on news articles:

| Attribute | Description |
| - | - |
| article_id | ID of news article. |
| article_url | Original URL of the news article. |
| article_title | Title of the news article (may include lead). |
| article_tags | Automatically generated tags for the main article text/body. |
| image_id | ID of news image (we provide a copy of the image). |
| image_url | Original URL of the news image. |

Furthermore, a folder 'newsimages' containing a copy of all news images is included.
The name of each JPG file corresponds to the article ID associated with each news article.

## Expected Submission

Image retrieval and generation have two subtasks each, a small one (using pre-determined article IDs that will be communicated in advance) and a large one (using randomly selected article IDs).
The articles in both the small and large tasks are part of the dataset shared with participants.
For more details, please see the [Task Overview Paper](TBD).

You must provide a ZIP file [group_name].zip that is structured as follows:

[group_name] + _ + [approach_name] / [article_id] + _ + [group_name] + _ + [approach_name].png

Use the group name with which you have registered for the task.
For each submitted approach/run, please provide a **unique name**.
Your submission can include multiple approaches.
Each approach must include **precisely one** image recommendation for a given article ID.

### Example Group Submission

Below is the folder structure of an example ZIP file for the group 'UnstableOsmosis':

    UnstableOsmosis_Submission.zip
    |_ UnstableOsmosis_FLUX
    |  |_ 117_UnstableOsmosis_FLUX.png
    |  |_ …
    |_ UnstableOsmosis_OpenCLIP
    |  |_ 117_UnstableOsmosis_OpenCLIP.png
    |  |_ …
    |_ …


### Required Image Format

The image format must be PNG, with target dimensions of 460x260 pixels (landscape orientation).
This applies to both generated and retrieved images.
If you generate the images with tools like ComfyUI and you edit them afterwards (e.g., for cropping), make sure the workflow **remains** embedded.

### Complete Email Submission

You will need to hand in your submissions by the deadline indicated below.
Do that by sending an email to the address that shared the dataset download link with you.
It must include (1) your group name, (2) a link to download your image submissions, and (3) links to the documented code of your workflow (e.g., a link to a GitHub repository with a notebook and/or a collection of scripts).
(Please note that this is something separate from the Working Notes Paper.)

## Online Evaluation

Taking part in the online evaluation event is mandatory.
During the evaluation, participating teams rate the image recommendations of other teams.
To do that, they are being presented with a news headline and two image recommendations.
They then need to select which image is more fitting.
These ratings are used to calculate an overall ranking of images for each article.
The average rank of team submissions across the featured item pool then determines the overall winner of the challenge.

## Working Notes Paper

As part of the challenge submission, each team is required to write a separate **Working Notes Paper** that documents and outlines their approach.
Please look at the [online paper template](https://drive.google.com/drive/folders/1DNhxIeACfsmg6rrdgQZ22BbRtYE8ioYI) for more information.

We encourage open and reproducible science.
We ask each team to share their codebase and workflows.
Please use the examples in the [designated folder](https://github.com/Informfully/Challenges/tree/main/newsimages26/workflows) to structure your code and make a pull request to contriubte your workflow.

Furthermore, we ask each group to include and refer to the following papers in their Working Notes Paper:

* [NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles](https://github.com/Informfully/Challenges/blob/main/documents/newsimages_task_overview_paper.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2025 Workshop, 2025.

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

* Challenge announcement: January
* Release development dataset: January
* Release test dataset: April
* Online evaluation: May (online)
* Evaluation feedback: May (AoE)
* Working Notes Paper submission: May*
* Review deadline: June**
* Camera-ready deadline: June (AoE).
* MediaEval workshop: June (more information on the [registration website](TBD), in-person or online attendance required).

(*) We provide you with a review/feedback for your paper within one week of submission.
Afterwards, you then have another week to prepare the camera-ready revision (exact deadlines will be communicated by the MediaEval organizers).
Please note that your paper should include a results section.
It is based on your performance in the online evaluation.
The necessary information for this part will be forwarded to you after the evaluation event has concluded.
The Working Notes Paper **must** describe the workflows for your submissions.
It **may** include complementary and/or alternative approaches that you tested.
We also encourage all teams to write a separate "Quest for Insight" paper if there are interesting findings you would like to share and discuss with (for more information, see "Quest for Insight" in our challenge overview: <https://multimediaeval.github.io/editions/2025/tasks/newsimages>).

(**) We will notify each team once their paper has been reviewed; please make the necessary changes and upload a camera-ready version within one week.

## Resources

* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Yahoo-Flickr Creative Commons 100 Million (YFCC100M)](https://www.multimediacommons.org)
