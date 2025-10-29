import json
import os
from tqdm import tqdm
import pandas as pd

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, logging
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")

logging.set_verbosity_error()
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model.cuda()



avsyear = 'mediaeval25'  # 'mediaeval25' 'mediaeval25_imgCapti' 'mediaeval25_summ'  'mediaeval25_images'
ROOT_DIR = "/m2/YFCC100M/images"
LISTS_DIR = f"../data/{avsyear}_combination_results_4000/5_5_1_20_30"

for inpath in os.listdir(LISTS_DIR):
    if 'results' in inpath:
        continue
    if '.html' in inpath:
        continue
    topicspath = '../data/newsimages_25_v1.1/subset.csv'

    df = pd.read_csv(topicspath, sep=",")

    df["article_text"] = ""
    lineList=[]
    topics = {}
    for idx, row in df.iterrows():
        lineList.append(row["article_title"])
        topics[str(row["article_id"])] = row["article_title"].strip()

    shots = {qid: [] for qid in topics}
    with open(os.path.join(LISTS_DIR, inpath), "r") as f:
        for line in f:
            args = line.split()
            if "msr-vtt" in inpath:
                shots[args[0]].append(args[2])
            else:
                shots[args[0]].append(args[2])

    for qid, topic in topics.items():
        results = {}
        for shot in tqdm(shots[qid][:300]):
            args = shot.split("_")
            basep = os.path.join(ROOT_DIR)

            if "msr-vtt" in inpath:
                p = "file://" + os.path.join(basep, "{}.jpg".format(frame_num))
            else:
                p = (
                    "file://"
                    + os.path.join(basep, shot)
                    + ".jpg"
                )
            prompt = "How relevant is this image to the following article title: {}? Answer with a single number 1-10.".format(
                topic
            )

            msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": p,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            seg = process_vision_info(msgs)[0]

            inputs = processor(
                text=[prompt], images=seg, padding=True, return_tensors="pt"
            ).to("cuda")

            scores = model.generate(**inputs, do_sample=False, max_new_tokens=200)[0][
                len(inputs.input_ids[0]) :
            ]
            out = processor.batch_decode([scores], skip_special_tokens=True)[0]
            results[shot] = out

        outname = os.path.splitext(inpath)[0]
        outdir = f"../data/rerank/results_{avsyear}"
        os.makedirs(outdir, exist_ok=True)
        with open("{}/greedy-{}-{}.json".format(outdir, outname, qid), "w") as f:
            json.dump(results, f, indent=2)
