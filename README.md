# Artist Classfication Ai
## Deployment
https://huggingface.co/spaces/Bravefe/Artist_Classification
## Blog
https://medium.com/@arecracerpkppkp/artist-classfication-ai-e3204129cba0


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## code
import shutil
from google.colab import files

# Define the source directory (folder) you want to zip
source_dir = "/content/Data"

# Define the name of the zip file you want to create
zip_filename = "/content/ZipData.zip"

# Zip the folder
shutil.make_archive(zip_filename.split(".")[0], 'zip', source_dir)

# Provide a download link for the zip file
files.download(zip_filename)

import zipfile

# Define the path to the zip file you want to extract
zip_filename = "/content/Data.zip"

# Define the directory where you want to extract the contents
extract_path = "/content/Data"

# Create the extraction directory if it doesn't exist
import os
os.makedirs(extract_path, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the contents of the extraction directory
extracted_files = os.listdir(extract_path)
print("Extracted files and directories:")
print(extracted_files)

import sys, os, requests, shutil
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
print("Starting...")

def scrape(tag,max_page,start_page,folder):
    print("scraping...")
    MAX_PAGE_SEARCH = max_page
    page = start_page

    for ctr in range(1, MAX_PAGE_SEARCH + 1):

        if tag == 'Draw':
          req = Request('https://danbooru.donmai.us/posts?page=' + str(page))
        else:
          req = Request('https://danbooru.donmai.us/posts?page=' + str(page) +
                      '&tags=' + space_to_underscore(tag), headers={'User-Agent': 'Mozilla/5.0'})

        webpage = urlopen(req).read()

        spage = soup(webpage, "html.parser")
        images = spage.findAll("article")
        print("(Found " + str(len(images)) + " images)")

        folder_path = os.path.join("/content/Data/images/"+folder, tag)
        # folder_path = os.path.join("/content/Data/images/"+folder, "Draw")

        os.makedirs(folder_path, exist_ok=True)

        print('downloading images...')

        for image in images:
            print(image.img['src'])
            image_url = image.img['src']
            r = requests.get(image_url, stream=True, headers={'User-agent': 'Mozilla/5.0'})

            if r.status_code == 200:
                filename = os.path.join(folder_path, image_url.split('/')[-1])
                print(filename)

                with open(filename, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
        page += 1

def space_to_underscore(string_):
    return "_".join(string_.split())

data = ["mika_pikazo","lack","shirabi","chigusa_minori","qqqrinkappp","aono3","criis-chan","komatsu eiji","kukie-nyan","konbu_wakame","donguri_suzume","modare","scottie_(phantom2)","xiujia_yihuizi","chigusa_minori","fujima_takuya","jonsun","komone_ushio","qys3","efe","ririko_(zhuoyandesailaer)","apple_caramel"]

for i in data:
  for j in range(1, 14, 3):
      scrape(i,1,j,"test")
  for j in range(2, 15, 3):
      scrape(i,1,j,"valid")
  for j in range(2, 16, 3):
      scrape(i,1,j,"train")

  scrape(i,15,13,"train")

  !pip install -q fastbook==0.0.29
from fastbook import 

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(valid_name='valid'),
    get_y=parent_label,
    batch_tfms=aug_transforms(size=250),
    item_tfms=Resize(250)
)
dls = dblock.dataloaders('Data/images/', bs=50)

dls.train.show_batch(max_n=9,nrows=3) #180

dls.train.show_batch(max_n=9,nrows=3) #512

learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fine_tune(epochs=30, freeze_epochs=0, base_lr=2e-3)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10))

interp.print_classification_report()

learn.export()
path = Path()
path.ls(file_exts='.pkl')

learn_inf = load_learner(path/'export.pkl')

from IPython.display import display
import ipywidgets as widgets
from PIL import Image

def upload_and_predict(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(512, 512))
    _, _, probs = learn_inf.predict(img)
    sorted_probs = sorted(zip(learn_inf.dls.vocab, map(float, probs)), key=lambda p: p[1], reverse=True)
    predictions_str = '; '.join(f'{label}: {prob:.5f}' for label, prob in sorted_probs)
    lbl_pred.value = f'Predictions: {predictions_str}'

btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()

btn_upload.observe(upload_and_predict, names=['data'])

display(btn_upload)
display(out_pl)
display(lbl_pred)

!git clone https://huggingface.co/spaces/Bravefe/Artist_Classification

%cd Artist_Classification

!ls

!pip install -r requirementsl.txt -q

learn_inf = load_learner('/content/ai_builder1.1.pkl')

!pip install gradio -q

import gradio as gr
import os
import pickle


def greet(image):
    pred, pred_idx, probs = learn_inf.predict(image)
    txt = f'Prediction: {pred} Probability: {probs[pred_idx]:.04f}'
    # sorted_probs = sorted(zip(learn_inf.dls.vocab, map(float, probs)), key=lambda p: p[1], reverse=True)
    # predictions_str = '; '.join(f'{label}: {prob:.5f}' for label, prob in sorted_probs)
    return txt

# def greet(name):
#     return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="image", outputs="label")
iface.launch(share=True)
