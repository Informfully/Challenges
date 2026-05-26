from datetime import datetime
import os

MODEL_ID_LLM = "Qwen/Qwen3-VL-8B-Instruct"

SET = 'test' # options: 'dev', 'test'
RERUN_INITIAL_PROMPT = False
THRESHOLD = 4
ITERATIONS = 6
IMG_GEN_MODELS = ['zt'] # options: 'zt', 'sdnq', '2512' 
CRITIC_FEEDBACK = True


if SET == 'dev':
    csv_file = "newsimages_test_and_evaluation_26_v1.0/news_articles_2025_updated.csv"
elif SET == 'test':
    csv_file =  "newsimages_test_and_evaluation_26_v1.0/news_articles_test.csv"

RUN_ = f"{SET}_{'_'.join(IMG_GEN_MODELS)}_thr_{THRESHOLD}_it_{ITERATIONS}_crFdn_{CRITIC_FEEDBACK}"
RUN_ID = f"{RUN_}{os.path.sep}{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
RUN_ID = os.path.join("Results", RUN_ID)
os.makedirs(RUN_ID, exist_ok=True)
