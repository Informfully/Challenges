from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd

def summarize(text):
    prompt = f"""You are a news assistant. Read the following article and generate a very short headline
     (like a newspaper title) in a way that it could be associated with a relevant image. Provide the summary and the image caption.

Keep it under 12 words.

    Article:
    {text}

    Summary:"""

    result = summarizer(prompt)[0]["generated_text"]
    return result

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 is best for LLaMA
    bnb_4bit_compute_dtype="float16"  # or "bfloat16" if your GPU supports it
)

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    # load_in_8bit=True # saves memory with bitsandbytes
)

summarizer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9
)

mediaeval_text_file = '../data/newsimages_25_v1.1/newsarticles_with_text.csv'


df = pd.read_csv(mediaeval_text_file, sep=",")

lineList = []
summaries = []
captions = []
for idx, row in df.iterrows():
    lineList.append(row["article_text"])
    text = row["article_text"]
    summ = summarize(text)
    summary = summ.split("Summary:")[-1].strip().split("\n\n")[0].strip()
    print(summary)

    caption = summ.split("Image Caption:")[-1].strip().split("\n\n")[0].strip()
    print(caption)
    summaries.append(summary)
    captions.append(caption)

df["summary"] = summaries
df["caption"] = captions


# saving the dataframe
mediaeval_text_file_save = '../data/newsimages_25_v1.1/newsarticles_with_text_summ_capt.csv'
df.to_csv(mediaeval_text_file_save, index=False)

