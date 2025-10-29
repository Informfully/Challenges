import os
import json
import numpy as np
from PIL import Image
import pandas as pd

def check(resultFile, pattern):
    with open(resultFile) as f:
        datafile = f.readlines()
    for line in datafile:
        if pattern in line:
            print(line.rstrip("\n\r"))


def copy_resize_rename_image(input_path, output_folder, new_name, size=(460, 260)):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the image
    with Image.open(input_path) as img:
        # Resize to exact size (force resize)
        resized_img = img.resize(size, Image.LANCZOS)

        # Ensure .png extension
        if not new_name.lower().endswith(".png"):
            new_name += ".png"

        output_path = os.path.join(output_folder, new_name)

        # Save as PNG
        resized_img.save(output_path, "PNG")


year = 'mediaeval25_summ'  # 'mediaEval2025' 'mediaeval25_imgCapti' 'mediaeval25_summ' 'mediaeval25_images'
llm_quest = 'HowRelevant10' # RateCorrect10 - ContainsYesNo - HowRelevant10 VideoHowRelevant-4frames
llm_vers = 'q2_5'
model = 'rephrase_ten_no_image_video_5_5_1_20_30_2_15_5_30_10_2000'

if year == 'mediaEval2025':
    year_llm = 'mediaeval25'
else:
    year_llm = year

w1 = 0.8 # Weight for relevance ranking
w2 = 1.0 - w1  # Weight for score annotations

run = f"greedy-{year_llm}_ALL"
run_id = ''
if year == 'mediaEval2025':
    if w1==1.0:
        run_id = 'RUN_5'
    else:
        run_id = 'RUN_1'
elif year == 'mediaeval25_imgCapti':
    run_id = 'RUN_4'
elif year == 'mediaeval25_summ':
    run_id = 'RUN_3'
elif year == 'mediaeval25_images':
    run_id = 'RUN_2'

group_name = 'CERTH-ITI'
outFolder = f'Submission/{group_name}/RET_{run_id}_SMALL/'
os.makedirs(outFolder, exist_ok=True)

queries_file = 'rerank/' + year + '.topics.txt' # 'avs.progress.22.23.24.topics.txt' 'tv22.avs.topics.txt'

queries_file = '../data/newsimages_25_v1.1/subset.csv'

df = pd.read_csv(queries_file, sep=",")

df["article_text"] = ""
lineList = []
topics = {}
image_urls = {}
for idx, row in df.iterrows():
    lineList.append(row["article_title"])
    topics[str(row["article_id"])] = row["article_title"].strip()
    image_urls[str(row["article_id"])] = row["image_url"].strip()

rerank_folder = f"../data/rerank/results_{llm_vers}/{model}"

save_folder = f"rerank/Final_reranking_{llm_vers}/{year}"
os.makedirs(save_folder,exist_ok=True)


LISTS_DIR = f"../data/{year_llm}_combination_results_4000/5_5_1_20_30/"

results_file = f"{LISTS_DIR}/{year_llm}_ALL.txt"

file1 = open(results_file, 'r')
Lines = file1.readlines()
Lines = [l.replace('  ', ' ') for l in Lines]

# read queries
# queries_file = 'tv21.avs.topics.txt'
file2 = open(queries_file, 'r')
queries = topics

# find unique queries
ids=[]
id_unique=[]
for li in Lines:
    id = li.split(' ')[0]
    ids.append(id)
    if id not in id_unique:
        id_unique.append(id)


newrresults_file =  f"{year_llm}_{run}_{model}_lateFusionTest_rerank.txt"
fh = open(newrresults_file, 'w')

correct_nums = []
incorrect_nums = []

thress = 5
imag_folder = '/m2/YFCC100M/images/'
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Articles with Images</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 20px;
      background: #f9fafb;
      color: #333;
    }}
    h1 {{
      text-align: center;
      margin-bottom: 20px;
      color: #222;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}
    th, td {{
      padding: 12px;
      text-align: center;
      vertical-align: middle;
    }}
    th {{
      background-color: #4f46e5;
      color: white;
      font-weight: 600;
    }}
    tr:nth-child(even) {{
      background-color: #f3f4f6;
    }}
    tr:hover {{
      background-color: #e0e7ff;
    }}
    img {{
      max-width: 120px;
      max-height: 120px;
      border-radius: 8px;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    img:hover {{
      transform: scale(1.1);
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    a {{
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <h1>Articles with Retrieved and Stored Images</h1>
  <table>
    <thead>
      <tr>
        <th>Article ID</th>
        <th>Article Title</th>
        <th>Stored Image</th>
        <th>Retrieved Image 1</th>
        <th>Retrieved Image 2</th>
        <th>Retrieved Image 3</th>
        <th>Retrieved Image 4</th>
        <th>Retrieved Image 5</th>
        <th>Retrieved Image 6</th>
        <th>Retrieved Image 7</th>
        <th>Retrieved Image 8</th>
        <th>Retrieved Image 9</th>
        <th>Retrieved Image 10</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""

rows = ""
num = 0

for quer, q in  enumerate(id_unique):
    res_list = [i for i in range(len(ids)) if ids[i] == q]

    query = queries[q].strip()
    shots = []
    for shot in res_list[:]:
        shots.append(Lines[shot].split(' ')[2])

    print(query)

    ranks_list = []
    scores_list = []
    video_ids = []

    # If the JSON is in a file, load it from the file
    # with open('rerank/lists/results_q2_5/' + llm_quest + '/' + year + '/' + run + '/' + year_llm + '_' + run + '-' + q[1:] + '.json', 'r') as file:
    #     data = json.load(file)

    with open(f'../data/rerank/results_{year}' + '/' + run + '-' + q + '.json', 'r') as file:
        data = json.load(file)

    for i, shot in enumerate(shots):
        video_ids.append(shot)
        # Access the value of <shot>
        item = data.get(shot)

        ranks_list.append(i+1)
        if item is not None:
            if item.isnumeric():
                scores_list.append(int(item))
            else:
                if ('The video is very' in item):
                    scores_list.append(10)
                else:
                    scores_list.append(0)
        else:
            scores_list.append(0)


    ranks = np.array(ranks_list)
    scores = np.array(scores_list)



    print()

    # Step 1: Normalize the relevance ranks (0 to 1)
    normalized_ranks = 1 - (ranks - 1) / (len(ranks) - 1)

    # Step 2: Normalize the scores (1 to 10 -> 0.1 to 1)
    normalized_scores = scores / 10.0

    # Step 3: Compute the combined score using a weighted sum
    combined_scores = w1 * normalized_ranks + w2 * normalized_scores
    # Step 4: Re-rank based on combined score
    sorted_indices = np.argsort(combined_scores)[::-1]

    retrieved_cells = ""
    i=0
    for ss in sorted_indices[:1000]:

        if i < 10:
            imgID = video_ids[ss]
            img_path = os.path.join(imag_folder, f"{imgID}.jpg")
            retrieved_cells += f'<td><a href="{img_path}" target="_blank"><img src="{img_path}" alt="Retrieved Image {i + 1}"></a></td>'
            if i == 0:
                print("Submission preparation")
                # [article_id] + _ + [group_name] + _ + [approach_name]
                new_name = f'{q}_{group_name}_{run_id}'
                copy_resize_rename_image(img_path, outFolder, new_name, size=(460, 260))
        i+=1

    html_row = f"""
      <tr>
        <td>{q}</td>
        <td>{topics[q]}</td>
        <td><a href="{image_urls[q]}" target="_blank"><img src="{image_urls[q]}" alt="Stored Image"></a></td>
        {retrieved_cells}

      </tr>
    """
    rows += html_row
    num += 1

html_content = html_template.format(rows=rows)
#
saveFile_mediaeval25 = LISTS_DIR + f'/{year}_' + f'{year}_ALL_rerank_w1_{w1}_w2_{w2}'
with open(saveFile_mediaeval25 + '.html', "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Beautiful HTML file created: {saveFile_mediaeval25}")
