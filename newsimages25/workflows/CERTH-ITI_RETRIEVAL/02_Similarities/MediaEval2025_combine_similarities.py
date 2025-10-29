import os
import numpy as np
from basic.bigfile import BigFile
import torch
import tqdm
import pickle
import pandas as pd


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm

class Dataset4DualEncoding(torch.utils.data.Dataset):
    def __init__(self, visual_feat, do_visual_feas_norm, video2frames=None):
        self.video2frames = video2frames
        self.do_visual_feas_norm = do_visual_feas_norm

        self.video_ids = [key for key in self.video2frames.keys()]
        self.visual_feat = visual_feat

        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            if self.do_visual_feas_norm:
                frame_vecs.append(do_L2_norm(self.visual_feat.read_one(frame_id)))
            else:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(np.array(frame_vecs))
        return frames_tensor, index, video_id

    def __len__(self):
        return self.length

def read_dict(filepath):
    with open(filepath, 'r') as f:
        dict_data = eval(f.read())
    return dict_data

def check(resultFile, pattern):
    with open(resultFile) as f:
        datafile = f.readlines()
    for line in datafile:
        if pattern in line:
            print(line.rstrip("\n\r"))

def process_combinatation(norm_array, year, pre_trained_model, video_ids, SAVE_DIR):
    pre_trained_model = pre_trained_model.replace('-', '_').replace('/', '_')

    if year == 'mediaeval25':
        queryfile = '../data/newsimages_25_v1.1/subset.csv'
        df = pd.read_csv(queryfile, sep=",")
        df["article_text"] = ""
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["article_title"])
    elif year == 'mediaeval25_summ':
        queryfile = '../data/newsimages_25_v1.1/subset_with_text_summ_capt.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["summary"])
    elif year == 'mediaeval25_imgCapti':
        queryfile = '../data/newsimages_25_v1.1/subset_with_text_summ_capt.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["caption"])
    elif year == 'mediaeval25_images':
        queryfile = '../data/newsimages_25_v1.1/subset_with_text_summ_capt.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["image_id"])


    saveFile_mediaeval25 = SAVE_DIR + f'/{year}_' + pre_trained_model
    mediaeval_eval(saveFile_mediaeval25, df, video_ids, norm_array, 4000)


def mediaeval_eval(saveFile_mediaeval25, df, VideoIDS, norm_array, num_videos=4000):
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
    f = open(saveFile_mediaeval25 + '.txt', "w")
    for _, article in df.iterrows():
        queryError = norm_array[:, num]
        scoresIndex = np.argsort(queryError)

        retrieved_cells = ""
        for i, ind in enumerate(scoresIndex[:]):
            imgID = VideoIDS[ind]
            img_path = os.path.join(imag_folder, f"{imgID}.jpg")
            if i < 10:
                retrieved_cells += f'<td><a href="{img_path}" target="_blank"><img src="{img_path}" alt="Retrieved Image {i+1}"></a></td>'

            f.write(str(article["article_id"]))
            f.write(' 0 ' + imgID + ' ' + str(i) + ' ' + str(10000 - i) + ' ITI-CERTH' + '\n')
            if i == num_videos:
                break

        html_row = f"""
          <tr>
            <td>{article["article_id"]}</td>
            <td>{article["article_title"]}</td>
            <td><a href="{article["image_url"]}" target="_blank"><img src="{article["image_url"]}" alt="Stored Image"></a></td>
            {retrieved_cells}
            
          </tr>
        """
        rows += html_row
        num += 1

    html_content = html_template.format(rows=rows)

    with open(saveFile_mediaeval25 + '.html', "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML file created: {saveFile_mediaeval25}")



def main():
    video_set = 'YFCC100M'
    year = 'mediaeval25'  # 'mediaeval25' 'mediaeval25_imgCapti' 'mediaeval25_summ'  'mediaeval25_images'

    type = 'sum' # 'sum' 'max' 'mean'
    weights = np.array([5, 5, 1, 20, 30])  # shape (5,)

    rer = "_".join(map(str, weights))
    SAVE_DIR = f"../data/{year}_combination_results_4000/{rer}/"
    os.makedirs(SAVE_DIR,exist_ok=True)

    # Separate npy_files into tv22 and tv23
    _folder = f'/home/aleventakis/PycharmProjects/AVS/data/{year}/'
    npy_files = [
        f"{_folder}{year}_YFCC100M_BLIP_2_itm_similarities.npy",
        f"{_folder}{year}_YFCC100M_BLIP_itc_similarities.npy",
        f"{_folder}{year}_YFCC100M_SLIP_similarities.npy",
        f"{_folder}{year}_YFCC100M_CLIP_similarities.npy",
        f"{_folder}{year}_beit3_L2_similarities.npy",
    ]
    # Determine pre_trained_model
    pre_trained_model = 'ALL'

    # Process files
    errors = []
    for npy_file in npy_files:
        error = np.load(npy_file)
        errors.append(error)
    errors = np.stack(errors, axis=2)
    if type == 'sum':
        sum_array_ = np.sum(errors * weights[None, None, :], axis=2)
    if type == 'mean':
        sum_array = np.mean(errors * weights[None, None, :], axis=2)
    elif type == 'max':
        sum_array = np.max(errors * weights[None, None, :], axis=2)
    norm_array = sum_array / np.linalg.norm(sum_array, axis=1, keepdims=True)

    # Save norm_array to npy file
    savefile = f"{_folder}{year}_{pre_trained_model}_L2_similarities.npy"
    np.save(savefile, norm_array)


    visual_feat_path = os.path.join('/m2/YFCC100M/YFCC100M_Features/'+ video_set + '/', 'FeatureData', 'beit3_base_patch16_384_coco_retrieval')
    visual_feats = BigFile(visual_feat_path)
    video2frames = read_dict(os.path.join(visual_feat_path, 'video2frames.txt'))


    dset = Dataset4DualEncoding(visual_feats, 1, video2frames=video2frames)
    data_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=128, shuffle=False, pin_memory=True,
                                              num_workers=5)
    #
    video_ids = []
    for i, (videos, idxs, vid_ids) in enumerate(tqdm.tqdm(data_loader)):
        video_ids.extend(vid_ids)
    with open('YFCC100M_video_ids.pkl', 'wb') as ff:
        pickle.dump(video_ids, ff)

    with open('YFCC100M_video_ids.pkl', 'rb') as ff:
        video_ids = pickle.load(ff)

    process_combinatation(norm_array, year, pre_trained_model, video_ids, SAVE_DIR)

if __name__ == "__main__":
    main()
