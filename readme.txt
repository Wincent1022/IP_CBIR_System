If use vsc:
   - create a new env in anaconda prompt first
   - command used:
      - conda create -n cbir_env python=3.10 -y
      - conda activate cbir_env
      - pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn pillow jupyter ipykernel
      - python -m ipykernel install --user --name cbir_env --display-name "Python (CBIR)"
      - conda init
      - pip install streamlit
      - pip install numpy opencv-python scikit-image scikit-learn pillow streamlit

-----------------
| Run Streamlit |
-----------------
In anaconda prompt run:
   - conda activate cbir_env
   - cd to entier folder and run:
      - streamlit run app.py

In VSC, Open terminal (cmd) and run:
   - conda activate cbir_env
   - streamlit run app.py