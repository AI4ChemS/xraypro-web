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
from pdf2image import convert_from_path

from xraypro.xrayRec import loadMulti
from xraypro.MOFormer_modded.dataset_multi import MOF_ID_Dataset_Multi
import requests

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

    recommendationClass['CH4 storage'] = "Promising" if uptakeHP*(1 + percentage/100) >= thresholds['CH4Storage'] else "Not promising"
    recommendationClass['H2 storage'] = "Promising" if h2Cap*(1 + percentage/100) >= thresholds['H2Storage'] else "Not promising"
    recommendationClass['Xe storage'] = "Promising" if xeUptake*(1 + percentage/100) >= thresholds['XeStorage'] else "Not promising"
    recommendationClass['DAC'] = "Promising" if logKH*(1 - percentage/100) >= thresholds['DAC'] else "Not promising"
    recommendationClass['Carbon capture'] = "Promising" if uptakeLP*(1 + percentage/100) >= thresholds['CCapture'] else "Not promising"
    recommendationClass['Band gap'] = "Promising" if bandGap*(1 - percentage/100) <= thresholds['BandGap'] else "Not promising"

    return recommendationClass

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
            st.write("Prediction: ", output)

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
st.write("If you wish to cite us, please use the BibTeX below:")