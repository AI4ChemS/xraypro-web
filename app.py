import streamlit as st
import torch
import pandas as pd
import numpy as np

from xraypro.xraypro import loadModel
from utils.transform_pxrd import transformPXRD
import matplotlib.pyplot as plt

from xraypro.MOFormer_modded.dataset_recc import MOF_ID_Dataset
from xraypro.MOFormer_modded.tokenizer.mof_tokenizer import MOFTokenizer

from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import tempfile
import os

from xraypro.xrayRec import loadMulti
from xraypro.MOFormer_modded.dataset_multi import MOF_ID_Dataset_Multi
import requests
from sklearn.manifold import TSNE
import pickle
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import KDTree

WEIGHTS_URL = "https://drive.google.com/drive/folders/1Yw_7y3NBrzjKt3H-jHRm7ZaCu0AZNjPA"

def load_model(model_path):
    if torch.cuda.is_available():
        model = loadModel(mode = 'None').regressionMode()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        model = loadModel(mode = 'None').regressionMode()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

def xy_to_vector(path_to_xy):
    data = np.loadtxt(path_to_xy, skiprows=1)
    print(data)
    x, y = data[:, 0], data[:, 1]
    concat_data = np.array([x, y])
    y_t = transformPXRD(concat_data, two_theta_bound=(0, 40))
    return y_t

def plot_pattern(path_to_xy):
    y_t = xy_to_vector(path_to_xy)

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 40, 9000), y_t)
    ax.set_xlabel('2THETA')
    ax.set_ylabel('Normalized intensity')
    return fig

def preprocess_input(path_to_xy, precursor):
    path_to_xy.seek(0)
    y_t = xy_to_vector(path_to_xy)
    EXP_D = {'XRD' : [],
             'MOFid' : [],
             'Placeholder Label' : []
             }
    
    EXP_D['XRD'].append(y_t)
    EXP_D['MOFid'].append(precursor)
    EXP_D['Placeholder Label'].append(1)

    EXP_DF = pd.DataFrame(EXP_D)
    INDEX_ARRAY_EXP = EXP_DF.index.to_numpy().reshape(-1, 1)

    resultEXP = np.concatenate([INDEX_ARRAY_EXP, EXP_DF.to_numpy()], axis = 1)

    
    tokenizer = MOFTokenizer("xraypro/MOFormer_modded/tokenizer/vocab_full.txt")

    EXP_loader = DataLoader(MOF_ID_Dataset(data = resultEXP, tokenizer = tokenizer), batch_size=32, shuffle = True
                    )
    
    return EXP_loader

def predict(model, loader):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    with torch.no_grad():
        model.eval()
        predictions_test_EXP = []
        actual_test_EXP = []
        index_test_EXP = []

        for bn, (corr_index, input1, input2, target) in enumerate(loader):
            # compute output
            input2 = input2.unsqueeze(1).to(device)
            input1 = input1.to(device)
            output = model(input2, input1)
            
            for i, j, k in zip(output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten(), corr_index.numpy().flatten()):
                predictions_test_EXP.append(i)
                actual_test_EXP.append(j)
                index_test_EXP.append(k)
        
        return predictions_test_EXP[0]
    
def load_multi(model_path):
    if torch.cuda.is_available():
        model = loadMulti(mode = 'None').regressionMode()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    else:
        model = loadMulti(mode = 'None').regressionMode()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

def preprocess_input_multi(path_to_xy, precursor):
    y_t = xy_to_vector(path_to_xy)
    EXP_D = {'XRD' : [],
             'MOFid' : [],
             'Uptake at high pressure' : [],
             'H2 Capacity' : [],
             'Xe Uptake' : [],
             'log(KH)' : [],
             'Uptake at low pressure' : [],
             'Band gap' : []
             }
    
    EXP_D['XRD'].append(y_t)
    EXP_D['MOFid'].append(precursor)
    EXP_D['Uptake at high pressure'].append(1)
    EXP_D['H2 Capacity'].append(1)
    EXP_D['Xe Uptake'].append(1)
    EXP_D['log(KH)'].append(1)
    EXP_D['Uptake at low pressure'].append(1)
    EXP_D['Band gap'].append(1)

    EXP_DF = pd.DataFrame(EXP_D)
    INDEX_ARRAY_EXP = EXP_DF.index.to_numpy().reshape(-1, 1)

    resultEXP = EXP_DF.to_numpy()
    #resultEXP = np.concatenate([INDEX_ARRAY_EXP, EXP_DF.to_numpy()], axis = 1)

    
    tokenizer = MOFTokenizer("xraypro/MOFormer_modded/tokenizer/vocab_full.txt")

    EXP_loader = DataLoader(MOF_ID_Dataset_Multi(data = resultEXP, tokenizer = tokenizer), batch_size=32, shuffle = True
                    )
    
    return EXP_loader

def recommend(model, loader):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    with torch.no_grad():
        predictions_test = []
        actual_test = []

        for bn, (input1, input2, target) in enumerate(loader):
            target = target[:, :6]
            input2 = input2.unsqueeze(1).to(device)
            input1 = input1.to(device)
            output = model(input2, input1)
            
            pred_temp, actual_temp = [], []
            for i, j in zip(output.cpu().detach().numpy(), target.cpu().detach().numpy()):
                pred_temp.append(i)
                actual_temp.append(j)
            
            predictions_test.append(pred_temp)
            actual_test.append(actual_temp)

        predictions_test = np.concatenate(np.array(predictions_test), axis = 0)
        actual_test = np.concatenate(np.array(actual_test), axis = 0)
    
    thresholds = yaml.load(open("weights/thresholds.yaml", "r"), Loader=yaml.FullLoader)

    recommendationClass = {'CH4 storage' : None,
                           'H2 storage' : None,
                           'Xe storage' : None,
                           'DAC' : None,
                           'Carbon capture' : None,
                           'Band gap' : None
                           }
    
    uptakeHP = predictions_test[0][0]
    h2Cap = predictions_test[0][1]
    xeUptake = predictions_test[0][2]
    logKH = predictions_test[0][3]
    uptakeLP = predictions_test[0][4]
    bandGap = predictions_test[0][5]

    percentage = 15

    recommendationClass['CH4 storage'] = f"Promising ({np.round(uptakeHP*(1 + percentage/100), 2)} mol/kg)" if uptakeHP*(1 + percentage/100) >= thresholds['CH4Storage'] else f"Not Promising ({np.round(uptakeHP*(1 + percentage/100), 2)} mol/kg)"
    recommendationClass['H2 storage'] = f"Promising ({np.round(h2Cap*(1 + percentage/100), 2)} g/L)" if h2Cap*(1 + percentage/100) >= thresholds['H2Storage'] else f"Not Promising ({np.round(h2Cap*(1 + percentage/100), 2)} g/L)"
    recommendationClass['Xe storage'] = f"Promising ({np.round(xeUptake*(1 + percentage/100), 2)} mol/kg)" if xeUptake*(1 + percentage/100) >= thresholds['XeStorage'] else f"Not Promising ({np.round(xeUptake*(1 + percentage/100), 2)} mol/kg)"
    recommendationClass['DAC'] = f"Promising ({np.round(logKH*(1 - percentage/100), 2)} logKH)" if logKH*(1 - percentage/100) >= thresholds['DAC'] else f"Not promising ({np.round(logKH*(1 - percentage/100), 2)} logKH)"
    recommendationClass['Carbon capture'] = f"Promising ({np.round(uptakeLP*(1 + percentage/100), 2)} mol/kg)" if uptakeLP*(1 + percentage/100) >= thresholds['CCapture'] else f"Not promising ({np.round(uptakeLP*(1 + percentage/100), 2)} mol/kg)"
    recommendationClass['Band gap'] = f"Promising ({np.round(bandGap*(1 - percentage/100), 2)} eV)" if bandGap*(1 - percentage/100) <= thresholds['BandGap'] else f"Not Promising ({np.round(bandGap*(1 - percentage/100), 2)} eV)"

    return recommendationClass

def load_space():
    with open('tSNE/all_outputs.pickle', 'rb') as handle:
        all_outputs = pickle.load(handle)
    with open('tSNE/all_actual.pickle', 'rb') as handle:
        all_actual = pickle.load(handle)
    with open('tSNE/all_regr.pickle', 'rb') as handle:
        all_regr = pickle.load(handle)
    with open('tSNE/all_di.pickle', 'rb') as handle:
        all_di = pickle.load(handle)
    with open('tSNE/all_rho.pickle', 'rb') as handle:
        all_rho = pickle.load(handle)
    with open('tSNE/all_en.pickle', 'rb') as handle:
        all_en = pickle.load(handle)
    
    thresholds = yaml.load(open("weights/thresholds.yaml", "r"), Loader=yaml.FullLoader)
    embeddings_tsne = np.load('tSNE/embedding_space.npy')

    badMOF_col = 'lightgrey'
    row_indices = {'darkturquoise' : [], 'teal' : [], 'mediumspringgreen' : [], 'magenta' : [], 'darkmagenta' : [], badMOF_col : []}

    for bn, row in enumerate(all_regr):
        if row[0] >= thresholds['CH4Storage']:
            #colors.append('red')
            row_indices['darkturquoise'].append(bn)

        if row[1] >= thresholds['H2Storage']:
            #colors.append('blue')
            row_indices['teal'].append(bn)

        if row[2] >= thresholds['XeStorage']:
            #colors.append('green')
            row_indices['mediumspringgreen'].append(bn)

        if row[4] >= thresholds['CCapture']:
            #colors.append('purple')
            row_indices['magenta'].append(bn)
        
        if row[3] >= thresholds['DAC']:
            #colors.append('orange')
            row_indices['darkmagenta'].append(bn)

        if (row[0] < thresholds['CH4Storage']) and (row[1] < thresholds['H2Storage']) and (row[2] < thresholds['XeStorage']) and (row[3] < thresholds['DAC']) and (row[4] < thresholds['CCapture']):
            row_indices[badMOF_col].append(bn)

    return embeddings_tsne, all_outputs, all_regr, row_indices, all_di, all_rho, all_en

def interpolate_property(embeddings_tsne, new_point, all_di):
    tree = KDTree(embeddings_tsne)
    distances, indices = tree.query(new_point, k=5)
    weights = 1 / (distances + 1e-8)
    new_point_di = np.sum(weights * all_di[indices]) / np.sum(weights)

    return new_point_di

def tSNE_embedding(model, loader):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    with torch.no_grad():
        predictions_test = []
        actual_test = []

        for bn, (input1, input2, target) in enumerate(loader):
            target = target[:, :6]
            input2 = input2.unsqueeze(1).to(device)
            input1 = input1.to(device)
            output = model.model(input2, input1)
            
            pred_temp, actual_temp = [], []
            for i, j in zip(output.cpu().detach().numpy(), target.cpu().detach().numpy()):
                pred_temp.append(i)
                actual_temp.append(j)
            
            predictions_test.append(pred_temp)
            actual_test.append(actual_temp)

        predictions_test = np.concatenate(np.array(predictions_test), axis = 0)
        actual_test = np.concatenate(np.array(actual_test), axis = 0)
    
    #predictions_test is the new point to be added
    _, all_outputs, all_regr, row_indices, all_di, all_rho, all_en = load_space()

    X_with_new = np.vstack([all_outputs, predictions_test])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_with_new)

    original_points = X_tsne[:-1]
    new_point_tsne = X_tsne[-1]
    badMOF_col = 'lightgrey'

    unique_colors = {'darkturquoise': 'CH$_4$ storage', 'teal': 'H$_2$ storage', 'mediumspringgreen': 'Xe storage', 
                 'magenta' : 'Carbon capture','darkmagenta' : 'DAC'} 
    fig = go.Figure()

    data = np.stack((all_di[row_indices[badMOF_col]], all_rho[row_indices[badMOF_col]], all_en[row_indices[badMOF_col]]), axis=1)
    fig.add_trace(go.Scatter(
        x=original_points[row_indices[badMOF_col], 0],
        y=original_points[row_indices[badMOF_col], 1],
        mode='markers',
        marker=dict(size=7, color=badMOF_col, opacity=0.3),
        hovertemplate=(
                f'Application: Nothing interesting<br>'
            'Pore Diameter: %{customdata[0]:.2f} Å<br>'
            'Metal EN: %{customdata[2]:.2f}<br>'
            'Density: %{customdata[1]:.2f} g/cm³<extra></extra>'
            ),
            customdata=data,
        name='Nothing interesting',
        showlegend=True
    ))

    # Add data points for other categories with jittered positions
    noise_std = 4
    for i, (colour, indices) in enumerate(row_indices.items()):
        if colour != badMOF_col:
            data = np.stack((all_di[row_indices[colour]], all_rho[row_indices[colour]], all_en[row_indices[colour]]), axis=1)

            fig.add_trace(go.Scatter(
                x=original_points[indices, 0] + np.random.normal(0, noise_std / (i + 1), len(indices)),
                y=original_points[indices, 1] + np.random.normal(0, noise_std / (i + 1), len(indices)),
                mode='markers',
                marker=dict(size=7, color=colour, symbol='square', opacity=1),
                hovertemplate=(
                f'Application: {unique_colors[colour]}<br>'
            'Pore Diameter: %{customdata[0]:.2f} Å<br>'
            'Metal EN: %{customdata[2]:.2f}<br>'
            'Density: %{customdata[1]:.2f} g/cm³<extra></extra>'
            ),
            customdata=data,
                name=unique_colors[colour],
                showlegend=True
            ))
    
    pore_diameter_interpolate = interpolate_property(embeddings_tsne=original_points, new_point = new_point_tsne, all_di = all_di) #use KDTree to interpolate
    density_interpolate = interpolate_property(embeddings_tsne=original_points, new_point = new_point_tsne, all_di = all_rho)

    star_point = new_point_tsne.copy()
    fig.add_trace(go.Scatter(
        x=[star_point[0]],
        y=[star_point[1]],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        hovertemplate=f'Interpolated Pore Diameter: {pore_diameter_interpolate:.2f} Å<br>'
        f'Interpolated Density: {density_interpolate:.2f} g/cm³<extra></extra>',
        name='Your MOF!'
    ))

    fig.update_layout(
        width=1200,  # 21 inches * 100 pixels/inch
        height=550,  # 13 inches * 100 pixels/inch
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),  # No margins
    )

    st.plotly_chart(fig)


st.title("XRayPro: Connecting metal-organic framework synthesis to applications with a self-supervised multimodal model")

st.markdown("""
### Authors
**Sartaaj Khan**<sup>1</sup>, **Seyed Mohamad Moosavi**<sup>1</sup>
#### Affiliations
<sup>1</sup> Department of Chemical Engineering and Applied Chemistry, University of Toronto
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Adjust the sidebar width */
    [data-testid="stSidebar"] {
        width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

button_style = """
    <style>
    .container {
        display: flex;
        flex-wrap: wrap;
        justify-content: start;
    }
    .button {
        display: inline-block;
        padding: 10px 20px;
        margin-right: 10px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        text-decoration: none;
        background-color: #333;
        color: white;  /* This ensures text color is white */
        border-radius: 20px;
        vertical-align: middle;
    }
    .button img {
        width: 20px;
        height: 20px;
        margin-right: 8px;
        vertical-align: middle;
    }
    .button:hover {
        background-color: #555;
    }
    /* Ensure the anchor tag inside the button has white text */
    a.button {
        color: white !important;
    }
    a.button:hover {
        color: white !important;
    }
    </style>
"""

st.markdown(button_style, unsafe_allow_html=True)

st.markdown("""
    <div class="container">
        <a href="https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6" class="button">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"> arXiv
        </a>
        <a href="https://shorturl.at/TXLu7" class="button">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg"> PDF
        </a>
        <a href="https://github.com/AI4ChemS/XRayPro" class="button">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png"> Code
        </a>
    </div>
""", unsafe_allow_html=True)

model_selection = st.sidebar.radio(
    "What property are you interested in?",
    ("CH$_4$ uptake at HP (mol/kg)", "CO$_2$ uptake at LP (mol/kg)", "Surface area (m$^2$/m$^3$)", "log(K$_H$) of CO$_2$", "Crystal density (g/cm$^3$)", "H$_2$ storage capacity (g/L)", "Band gap (eV)")
)

mode_selection = st.sidebar.radio(
    "Recommendation system?",
    ("Yes", "No")
)

weightNames = {"CH$_4$ uptake at HP (mol/kg)" : "ft_CH4_Uptake_HP.h5",
               "CO$_2$ uptake at LP (mol/kg)" : "ft_CO2_Uptake_LP.h5",
               "Surface area (m$^2$/m$^3$)" : "ft_SA.h5",
               "log(K$_H$) of CO$_2$" : "ft_logKH.h5",
               "Crystal density (g/cm$^3$)" : "ft_density.h5",
               "H$_2$ storage capacity (g/L)" : "ft_H2_val.h5",
               "Band gap (eV)" : "ft_BG.h5"
               }

if mode_selection == "No" or mode_selection == "None":

    if model_selection == "CH$_4$ uptake at HP (mol/kg)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)

    elif model_selection == "CO$_2$ uptake at LP (mol/kg)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)

    elif model_selection == "Surface area (m$^2$/m$^3$)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)

    elif model_selection == "log(K$_H$) of CO$_2$":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)

    elif model_selection == "Crystal density (g/cm$^3$)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)
    
    elif model_selection == "H$_2$ storage capacity (g/L)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)
    
    elif model_selection == "Band gap (eV)":
        st.write(f"You selected {model_selection}")
        modelPath = f"weights/{weightNames[model_selection]}"
        model = load_model(model_path=modelPath)

    uploaded_file = st.file_uploader("Upload your .xy file", type = ["xy"])
    input_precursor = st.text_input("Enter precursor")

    if uploaded_file and input_precursor:
        st.write("Your inputs are: \n")
        st.write(f"Precursor: {input_precursor}")
        fig = plot_pattern(uploaded_file)
        st.pyplot(fig)

        uploaded_file.seek(0)
        #input_precursor.seek(0)

        loader = preprocess_input(uploaded_file, input_precursor)    
        
        if st.button('Make prediction'):
            output = predict(model, loader)
            st.write(f"Prediction for {model_selection}: ", output)

elif mode_selection == "Yes":
    st.write("Recommendation system is active.")
    modelPath = f"weights/ft_all.h5"
    model = load_multi(model_path=modelPath)

    uploaded_file = st.file_uploader("Upload your .xy file", type = ["xy"])
    input_precursor = st.text_input("Enter precursor")

    if uploaded_file and input_precursor:
        st.write("Your inputs are: \n")
        st.write(f"Precursor: {input_precursor}")
        fig = plot_pattern(uploaded_file)
        st.pyplot(fig)

        uploaded_file.seek(0)
        #input_precursor.seek(0)

        loader = preprocess_input_multi(uploaded_file, input_precursor)

        if st.button('What applications are good for my MOF?'):
            recs = recommend(model, loader)
            st.write(recs)

            tSNE_embedding(model, loader)

zip_file_path = "examples/MOF5.zip"

# Read the zip file as binary data
with open(zip_file_path, "rb") as f:
    zip_data = f.read()

st.download_button(
    label="Try out MOF-5!",
    data=zip_data,
    file_name="MOF5.zip",  # Use the appropriate file name for download
    mime="application/zip"
)

st.title("Overview of the model")

OVERVIEW = """
From leveraging the global domain provided by the PXRD pattern and the local environment from the chemical precursors, we are able to make both geometric, chemistry-reliant and electronic property predictions of metal-organic frameworks (MOFs) with just these two inputs which are usually readily available to experimentalists. 
As the precursors are usually in a string notation (SMILES alongside the metal node), an encoder and tokenizer are built on top of a transformer, in which the self-attention mechanism is utilized when knowing the absolute and relative positions of each token. The transformer is ultimately able to embed the chemical precursor.
A convolutional neural network (CNN) is then used to embed the PXRD pattern (preprocessed with a Gaussian transformation and ranging from 0 to 40 degrees). The two embeddings of the chemical precursor and the PXRD are concatenated together, in which it is fed into a regression head for predictions. For data efficiency purposes and to 
improve the predictions of local properties, self-supervised learning was done against a crystal graph convolutional neural network (CGCNN), as it provides information on the local environments. These pretrained weights are used and finetuned for any task. 
"""

st.write(OVERVIEW)

pdf_path = "figures/Methods.png"
st.image(pdf_path, caption = 'Workflow', use_column_width = True)

st.title("Does this work on any PXRD pattern?")
DOES_IT_WORK = """
We have tested our model on both "unclean" computational model PXRDs and experimental PXRDs and found it to be quite robust! We took MOFs with missing hydrogen atoms, bounded and unbounded solvents from the Cambridge Structural Database (CSD) and evaluated these on our finetuned model on methane uptake at high pressure, and found that despite 
the PXRDs looking quite different from the "clean" versions from CoRE-MOF, it still gives relatively consistent predictions. Furthermore, experimental PXRDs (courtesy of Howarth et al. from Concordia and from a paper based on pyrene-based MOFs) were inputted into our model, with consistent predictions on the recommendation for the MOFs.
"""
st.write(DOES_IT_WORK)

pdf_path = "figures/CSDAssessment.png"
st.image(pdf_path, caption = 'CSD robustness assessment', use_column_width = True)

st.title("How does it compare to other models?")
pdf_path = "figures/spider_w_cgcnn_v2.png"
st.image(pdf_path, caption = 'Radar plot', use_column_width = True)

st.title("Citation")

bibtex_entry = """
@article{khan2024connecting,
  title = {Connecting metal-organic framework synthesis to applications with a self-supervised multimodal model},
  author = {Khan, Sartaaj Takrim and Moosavi, Seyed Mohamad},
  year = {2024},
  journal = {ChemRxiv},
  doi = {10.26434/chemrxiv-2024-mq9b4},
  url = {https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6},
  note = {Preprint, not peer-reviewed}
}
"""

st.write("If you wish to cite us, please use the BibTeX below:")
st.code(bibtex_entry, language="bibtex")
