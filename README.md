## Main
XRayPro is a multimodal model that accepts the powder x-ray diffraction (PXRD) pattern and chemical precursors to make a wide variety of property predictions of metal-organic frameworks (MOFs), while supplementing the most feasible applications per MOF. This is a tool that is motivated by accelerating material discovery. If you are interested in more details, supplementary links have been provided below.

[Code](https://github.com/AI4ChemS/XRayPro) | [Paper](https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6)

![Methods](https://github.com/user-attachments/assets/fb2256ba-64cb-4ab4-8391-a1909b1ef576)

## Installation and usage
The live Streamlit application is available here: <URL>.

If you wish to install this application and run it locally on your machine, please do:

```bash
git clone https://github.com/AI4ChemS/xraypro-web.git
cd go/to/xraypro-web
pip install -r requirements.txt
streamlit run app.py

#if your streamlit command is not running, use:
python -m streamlit run app.py
```

Note: When uploading your PXRD pattern (in the form of a .xy file), please ensure that the first line has the diffractometer configurations (radiation type, wavelength, etc.)

## Privacy when using web application
Our web app tool does **NOT** store any data that is inputted into the entry fields (there is no external database for this).

## Citation
If our work is used, please cite us with the following BibTeX entry:
```bibtex
@article{khan2025connecting,
  title={Connecting metal-organic framework synthesis to applications using multimodal machine learning},
  author={Khan, Sartaaj Takrim and Moosavi, Seyed Mohamad},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={5642},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
