## Main
XRayPro is a multimodal model that accepts the powder x-ray diffraction (PXRD) pattern and chemical precursors to make a wide variety of property predictions of metal-organic frameworks (MOFs), while supplementing the most feasible applications per MOF. This is a tool that is motivated by accelerating material discovery. If you are interested in more details, supplementary links have been provided below.

[Code](https://github.com/AI4ChemS/XRayPro) | [Paper](URL)

![Methods](https://github.com/user-attachments/assets/fb2256ba-64cb-4ab4-8391-a1909b1ef576)

## Demo on how to use XRayPro
Need to record the video.

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

## Citation
If our work is used, please cite us with the following BibTeX entry:
```bibtex
@article{XRayPro,
  title={Connecting metal-organic framework synthesis to applications with a self-supervised multimodal model},
  author={Sartaaj Khan and Seyed Mohamad Moosavi},
  journal={journalName},
  year={2024},
  volume={volumeName},
  pages={pagesOfJournal},
  publisher={publisherName}
}
