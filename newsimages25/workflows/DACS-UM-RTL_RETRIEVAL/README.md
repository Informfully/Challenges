# NewsImages Retrieval at MediaEval 2025

## Setup

Python version: 3.11

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tests

```
python -m pytest tests/
```

## Methodology and initial findings

Our methods are based on the Retrieve and Re-rank approach. First, we retrieve a subset of candidate images for each query, and then we re-rank them using various re-ranking strategies. Both steps are done with pre-trained models, so that our methods are fully zero-shot:
- **Retrieve**: Candidate generation is based on vision-language models (CLIP, SigLIP, and other backbones) embeddings and similarity search. Check `src/models/candidate_generators.py` for details.
- **Rerank**: Reranking is based on additional components, implemented in `src/models/reranking.py`. The re-ranking logic fuses initial retrieval scores with the re-ranking scores to define the final score used for ranking. The re-ranking logic is only applied to the subset of candidate images that were retrieved in the initial retrieval step.

Supported backbones are available in `src/models/configs.py`.
We implement four re-ranking strategies:
- **Identity**: No re-ranking. Purely based on the similarity search with backbone embeddings.
- **VLM Judge**: We use a VLM model to assess candidate images in terms of their quality, truth, potential biases, and clickbaitness: `Qwen/Qwen2-VL-7B-Instruct` for submissions.
- **LLM Rewriting**: We use an LLM model to rewrite more descriptive captions for the given query: `meta-llama/Meta-Llama-3-8B-Instruct` for submissions.
- **Aesthetics**: We use an aesthetic model to assess the aesthetics of the candidate images. The aesthetic model is adapted from [`LAION-AI/aesthetic-predictor`](https://github.com/LAION-AI/aesthetic-predictor/) for submissions.

Our results on the validation dataset show that the re-ranking strategies can improve the retrieval performance with the ground truth (editorially selected) thumbnails. The validation dataset is randomly sampled from the development dataset (10% of the development dataset -- 850 instances). We used images from this validation set as candidates in our experiments. Thus, we had 850 queries and 850 ground truth thumbnails, where each query had one ground truth thumbnail. 

We found that the re-ranking strategies can improve the retrieval performance when ground truth thumbnails are available. Initial findings:
- Depending on the backbone and re-ranking strategy, we found that our methods can lead to up to 3.5% improvement in Hits@1. 
- Re-ranking strategies are more effective for smaller and less performant backbones, whereas for larger backbones, the gains might be marginal.
- For all implemented re-ranking strategies, we found that re-ranking weights should be selected carefully. The optimal weights are backbone-dependent, but all optimal values are within the range of 0.1-0.2.

As previously reported in [1], editorially selected thumbnails are not always perceived as the best thumbnails for a given query, since alternative images can sometimes be judged as equally good or better fits. Motivated by this, we would like to investigate whether our re-ranking strategies can improve retrieval performance according to human judgment. Following the organizers’ suggestion, we use YFCC100M as a source of candidate images for the submission. For now, we utilize a subset of the dataset (first 250 shards) containing around 370,000 images to generate submissions with three backbones (`openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, `google/siglip2-so400m-patch14-384`). We also submit the results with `openai/clip-vit-large-patch14` using a larger subset of YFCC100M containing around 1.5 million images (300 shards). Note that re-ranking is applied with a weight of 0.2 for all re-ranking strategies, and the re-ranked retrieval result (top-1) might remain the same as the initial retrieval result (i.e., identity re-ranking).

We index all candidate images with `faiss` to retrieve the top-k most similar ones for each query and then apply our re-ranking strategies to the retrieved candidates. Using `faiss` will allow us to scale to larger subsets of YFCC100M in future work by leveraging approximate nearest neighbor search and product quantization techniques.

[1] Heitz, L., Bernstein, A., & Rossetto, L. (2024). An Empirical Exploration of Perceived Similarity between News Article Texts and Images. CEUR Workshop Proceedings (p. online). Presented at the MediaEval 2023 Multimedia Benchmark Workshop 2023, CEUR-WS. Retrieved from https://ceur-ws.org/Vol-3658/paper8.pdf

## Scripts

The scripts to prepare data,conduct experiments and run the inference with YFCC100M are available in the `scripts/` directory. 

Below are the instructions on how individual steps can be done manually.

## Data preparation

### NewsImages development dataset 

Download and unzip data (version 1.1):
```
mkdir data
wget -O data/newsimages_v1.1.zip https://seafile.ifi.uzh.ch/f/5b9c7fd921ff432f9120/?dl=1
unzip data/newsimages_v1.1.zip -d data/newsimages_v1.1/
```

### Preparing retrieval candidates from YFCC100M

#### Download data shards
The retrieval candidates are images from the [YFCC100M dataset](https://multimediacommons.wordpress.com/getting-started/#download) that contains 100 million images (more than 10 TiB of data) with metadata (about 60GB of data). The dataset is stored in AWS S3.

The data download is based on the open-source [`yfcc100m`](https://gitlab.com/jfolz/yfcc100m) package. We made a small modification to the original code in `./yfcc100m/yfcc100m/download.py` to allow specifying the number of shards to download.

0. Create and AWS account, [generate access and secret keys](https://aws.amazon.com/de/blogs/security/wheres-my-secret-access-key/), and use them to configure the AWS CLI:
    ```
    pip install awscli
    aws configure
    ```

1. Install the `yfcc100m` package dependencies:
    ```
    cd yfcc100m
    pip install -e .
    cd ..
    ```

2. Download the metadata split into shards (85 GB):
    ```
    mkdir data
    mkdir data/yfcc100m
    mkdir data/yfcc100m/meta
    python -m yfcc100m.yfcc100m.convert_metadata data/yfcc100m -o data/yfcc100m/meta
    ```

3. Download the images (100 first shards: ~16 GB):
    ```
    mkdir data/yfcc100m/images
    python -m yfcc100m.yfcc100m.download data/yfcc100m/meta/ -o data/yfcc100m/images/ --num_shards 100
    ```

4. Unzip the images:
    ```
    for file in data/yfcc100m/images/*.zip; do
        unzip $file -d data/yfcc100m/images/
    done
    ```

### Building search index with FAISS

The script for building local search index with `faiss`: `src/write_faiss_index.py`.

Example with CLIP base (`openai/clip-vit-base-patch32`) and default index (`flat_ip`):
```
python -m src.write_faiss_index \
    --data data/yfcc100m/images/data/images/ \
    --output faiss/shards_250 \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32;
```

<details>
<summary>Arguments</summary>

Required:
- `--data`: Path to your dataset containing images
- `--output`: Where to save the FAISS index file
- `--model_family`: Vision-language model family to use
- `--model_id`: Specific model ID from Hugging Face

Check available options for `--model_family` and `model_id` in `src/models/configs.py`.

Optional:
- `--batch_size`: Number of images to process at once (default: 32)
- `--index_type`: Type of FAISS index ('flat_ip' or 'hnsw') (default: flat_ip)
- `--m`: Number of connections per layer, only for 'hnsw' index (default: 32)
- `--device`: Device for generating embeddings ('cuda' or 'cpu') (default: cuda if available, else cpu)
</details>

## Models and pipelines:

Our retrieval pipelines are based on the retrieve and re-rank approach. Specifically, first we retrieve a set of candidate images for each query, and then we re-rank them using a re-ranking model:
- Candidate generation is based on vision-language models (CLIP, SigLIP, etc.) embeddings and similarity search. Check `src/models/candidate_generators.py` for details.
- Reranking is based on additional components, implemented in `src/models/reranking.py`. To skip re-ranking, use `--reranking_algorithm identity`.

### Retrieval models:
Supported retrieval backbones are available in `src/models/configs.py`.

### Development dataset

The development dataset contains 8,500 instances: article titles, metadata, and (ground-truth) thumbnail images of these articles were published with. We randomly split these instances into train/val/test sets as 80/10/10%. To run the zero-shot retrieval pipeline on the test set, use the `src/retrieval_pipeline_dataloader.py` script. 

By default, the retrieval pipeline on development data uses `wandb` to log performance metrics. Make sure to log in from the command line. Alternatively, you can run scripts with `--debug` flag to disable `wandb` logging.

Example with SigLIP:

```
python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family siglip \
    --model_id google/siglip-base-patch16-224
```

<details>
<summary>Arguments</summary>

- `--csv_path`: path to the CSV file with article metadata
- `--image_path`: path to the directory with article images
- `--model_family`: retrieval model family, e.g. "siglip"
- `--model_id`: specific model checkpoint ID
- `--data_split`: data split to use ("train", "val", "test")
- `--subset_split`: path to subset split file
- `--batch_size`: batch size for inference
- `--log_retrieved_images`: path to save retrieved images
- `--num_candidates`: number of candidates to retrieve
- `--reranking_algorithm`: reranking algorithm ("identity", "vlm_judge", "llm_rewriting", "aesthetics")
- `--reranking_config`: path to reranking config file
- `--reranking_vlm_prompt_path`: path to VLM reranking prompt file
- `--reranking_vlm_weight`: weight for VLM reranking
- `--reranking_llm_weight`: weight for LLM reranking
- `--reranking_llm_num_captions`: number of captions for LLM reranking
- `--reranking_aesthetic_local_path`: path to aesthetic model checkpoint
- `--reranking_aesthetic_model_name`: aesthetic model name
- `--reranking_aesthetic_weight`: weight for aesthetic reranking
- `--llm_captions_path`: path to LLM-generated captions JSON file
- `--debug`: enable debug mode
- `--seed`: random seed

</details>

To run benchmarking scripts without re-ranking (identity):

```
bash scripts/retrieval_holdout_identity.sh
```

### Generating queries with LLMs:
Currently, descriptive queries can be generated with Llama 3. The queries generated for submission are stored in `./data/captions/`.

It is possible to use different prompts and models with the following script:

- Prompts: `./configs/llm/prompts`
- Model and generation configs: `./configs/llm/generation`

Example script:
```
python -m src.generate_captions_from_titles \
    --csv_path ./data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --llm_config_path ./configs/llm/generation/llama38b_default.yaml \
    --system_prompt_path configs/llm/prompts/system_default.yaml \
    --user_prompt_path ./configs/llm/prompts/user_simple.yaml \
    --num_captions 5 \
    --output_path ./data/captions/llama38b_simple.json 
```

### Aesthetic re-ranking

Download aesthetic model weights (from [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor/tree/main)):
```
mkdir aesthetic_checkpoints
cd aesthetic_checkpoints/
wget https://github.com/LAION-AI/aesthetic-predictor/raw/refs/heads/main/sa_0_4_vit_l_14_linear.pth
wget https://github.com/LAION-AI/aesthetic-predictor/raw/refs/heads/main/sa_0_4_vit_b_32_linear.pth
cd ../
```


### Inference pipeline with FAISS index and queries

Example with CLIP base (`openai/clip-vit-base-patch32`), default index (`flat_ip`) created above, and identity re-ranking:
```
python -m src.retrieval_pipeline_inference \
   --faiss_index_path ./faiss/shards_250/openai/clip-vit-base-patch32/image_index.faiss \
   --image_paths_file ./faiss/shards_250/openai/clip-vit-base-patch32/image_paths.txt \
   --csv_path ./data/newsimages_v1.1/newsarticles.csv \
   --subset_split ./data/newsimages_v1.1/subset.csv \
   --model_family clip \
   --model_id openai/clip-vit-base-patch32 \
   --output_path ./results/inference/shards_250/identity/clip-vit-base-patch32 \
   --num_candidates 25;
```