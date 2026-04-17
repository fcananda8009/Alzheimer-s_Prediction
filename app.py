import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Alzheimer's Disease Classification | Hybrid ConvNeXt-Swin-LoRA", layout='wide')

st.markdown("<h1 style='text-align: center; color: gray;'>A Parameter-Efficient Hybrid ConvNeXt–Swin<br>for Alzheimer's Disease Classification</h1>", unsafe_allow_html=True)

page_style = '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
            '''

st.markdown(page_style,unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model = tf.keras.models.load_model("Trained_Model2.h5")

selected_tab = option_menu(
    menu_title = None,
    options = ['About Alzheimers', 'Model & Results', "How to Use", 'Alzheimers\'s Detection', "About Team"],
    icons = ['house-door', 'cpu', 'file-earmark-text', 'binoculars', 'people'],
    menu_icon = 'cast',
    default_index = 0,
    orientation = 'horizontal',
    styles={
        "icon": {"font-size": "18px"},
        "nav-link": {"font-size": "16px"}
    }
)



if selected_tab == 'Alzheimers\'s Detection':

    col1,col2,col3,col4,col5 = st.columns(5)
    with col2:
        st.markdown("<h5 style='text-align: center; color: gray;'>Normal Brain</h5>", unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b2/MRI_of_Human_Brain.jpg")

    st.write("##")
    data = st.file_uploader("Upload an MRI Image of the brain",type = ['png','jpeg','jpg'])
    if data:
        with col4:
            st.markdown("<h5 style='text-align: center; color: gray;'>Uploaded Image</h5>", unsafe_allow_html=True)
            st.image(tf.keras.utils.load_img(data, target_size = (1433,1534)))
    c1,c2,c3,c4,c5 = st.columns(5)
    with c3:
        butt = st.button("Click to check")
    if butt:
        if data:
            img = tf.keras.utils.load_img(data, target_size = (128,128),color_mode='rgb')
            img = tf.convert_to_tensor(img)
            pred = np.argmax(tf.nn.softmax(model.predict(tf.expand_dims(img, 0))[0]))
            with open("classes.pkl","rb") as f:
                d = pickle.load(f)
            st.info("The model predicts the image as "+d[pred])



elif selected_tab == 'About Alzheimers':
    st.markdown("<h6 style='text-align: center; color: gray;'>Alzheimer's disease (AD) is a neurodegenerative disease that usually starts slowly and progressively worsens, and is the cause of 60–70% of cases of dementia.\
         The most common early symptom is difficulty in remembering recent events.\
         As the disease advances, symptoms can include problems with language, disorientation (including easily getting lost), mood swings, loss of motivation, self-neglect, and behavioural issues.\
         As a person's condition declines, they often withdraw from family and society. Gradually, bodily functions are lost, ultimately leading to death.\
         Although the speed of progression can vary, the typical life expectancy following diagnosis is three to nine years.<h6>", unsafe_allow_html=True)
    
    c1,c2= st.columns([2,2])
    with c1:
        st.write("##")
        st.image("https://www.jax.org/-/media/AEC74EFDF0234A03AA44A4D461BC5E81.jpg",width = 500)
    with c2:
        # st.markdown(br,unsafe_allow_html=True)
        st.write("##")
        st.markdown("<h6 style='text-align: center; color: gray;'>The cause of Alzheimer's disease is poorly understood.\
                 There are many environmental and genetic risk factors associated with its development. \
                 The strongest genetic risk factor is from an allele of APOE. Other risk factors include a history of head injury, clinical depression, and high blood pressure.\
                 The disease process is largely associated with amyloid plaques, neurofibrillary tangles, and loss of neuronal connections in the brain.\
                 A probable diagnosis is based on the history of the illness and cognitive testing, with medical imaging and blood tests to rule out other possible causes.\
                 Initial symptoms are often mistaken for normal brain aging.\
                 Examination of brain tissue is needed for a definite diagnosis, but this can only take place after death.\
                 Good nutrition, physical activity, and engaging socially are known to be of benefit generally in aging, and may help in reducing the risk of cognitive decline and Alzheimer's.\
                <h6>", unsafe_allow_html=True)
    c1,c2 = st.columns([2,2])
    with c1:
        st.image("https://www.mycirclecare.com/wp-content/uploads/2018/06/Effect-of-Alzheimer-by-Stages.jpg", width = 500)

    with c2:
        st.markdown("<h6 style='text-align: center; color: gray;'>No treatments can stop or reverse its progression, though some may temporarily improve symptoms. Affected people become increasingly reliant on others for assistance, often placing a burden on caregivers.\
            The pressures can include social, psychological, physical, and economic elements. Exercise programs may be beneficial with respect to activities of daily living and can potentially improve outcomes.\
            Behavioural problems or psychosis due to dementia are often treated with antipsychotics, but this is not usually recommended, as there is little benefit and an increased risk of early death.\
            As of 2020, there were approximately 50 million people worldwide with Alzheimer's disease. It most often begins in people over 65 years of age, although up to 10% of cases are early-onset impacting those in their 30s to mid-60s. It affects about 6% of people 65 years and older, and women more often than men.\
            The disease is named after German psychiatrist and pathologist Alois Alzheimer, who first described it in 1906. Alzheimer's financial burden on society is large, with an estimated global annual cost of US$1 trillion. It is ranked as the seventh leading cause of death in the United States.\
            <h6>", unsafe_allow_html=True)
    
    st.markdown("<h6 style='text-align: center; color: white;'>This information was made available using Google Search and Wikipedia</h6>", unsafe_allow_html=True)


elif selected_tab == 'Model & Results':
    st.markdown("### Proposed Architecture: Hybrid ConvNeXt–Swin Transformer with LoRA")
    st.markdown("""
    Our model is a **dual-stream hybrid architecture** that fuses:
    - **ConvNeXt-Tiny (CNN branch):** Extracts local spatial features — captures fine-grained morphological patterns such as hippocampal atrophy, tissue-level changes, and ventricular boundary delineations.
    - **Swin-Tiny Transformer with LoRA (Transformer branch):** Models long-range global dependencies through shifted-window self-attention, capturing cortical atrophy patterns and inter-regional connectivity changes.

    Both branches produce a **768-dimensional feature vector**, which are concatenated (late fusion) into a **1536-D unified representation**, then classified through a shared head:
    `Linear(1536→512) → GELU → Dropout(0.3) → Linear(512→4)`
    """)

    st.write("##")

    # Architecture diagram placeholder — user will add image manually
    st.markdown("#### Model Architecture")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("imgs/hybrid_convnext_swin_lora_pipeline_vertical (1).png", caption="Hybrid ConvNeXt–Swin–LoRA Architecture", use_container_width=True)

    st.write("##")

    st.markdown("#### LoRA (Low-Rank Adaptation) Configuration")
    lora_data = {
        "Parameter": ["Rank (r)", "Alpha (α)", "Target Modules", "LoRA Dropout", "Bias"],
        "Value": ["8", "32", "query, key, value, dense", "0.1", "None"],
        "Description": [
            "Rank of the low-rank decomposition matrices",
            "Scaling factor; effective scale = α/r = 4.0",
            "Attention projection matrices adapted by LoRA",
            "Dropout applied to LoRA outputs for regularisation",
            "No bias parameters added to LoRA layers"
        ]
    }
    st.table(lora_data)

    st.markdown("""
    > **Key Insight:** LoRA rank r=8 was identified as optimal through a systematic ablation study (r ∈ {1, 4, 8, 16}).
    > It achieves **97.66% accuracy** with only **2.26% trainable parameters** (636K out of 28M) on the standalone Swin backbone —
    > surpassing full fine-tuning by +2.19% while training **43× fewer parameters**.
    """)

    st.write("##")

    # LoRA Rank Ablation
    st.markdown("#### LoRA Rank Ablation Study")
    c1, c2 = st.columns(2)
    with c1:
        ablation_data = {
            "LoRA Rank": ["r=1", "r=4", "r=8 ★", "r=16", "Full FT"],
            "Trainable Params": ["79K", "318K", "636K", "1.27M", "27.52M"],
            "Train %": ["0.29%", "1.14%", "2.26%", "4.42%", "100%"],
            "Accuracy": ["89.26%", "96.79%", "97.66%", "97.34%", "95.57%"],
            "ECE ↓": ["0.0717", "0.0253", "0.0185", "0.0208", "0.0362"]
        }
        st.table(ablation_data)
    with c2:
        st.image("imgs/rankvsece.jpg", caption="LoRA Rank vs Accuracy & ECE", use_container_width=True)

    st.write("##")

    # Model Performance Comparison
    st.markdown("#### Model Performance Comparison")
    perf_data = {
        "Model": [
            "EfficientNet-B0", "ResNet-50", "ViT-LoRA r=16", "ConvNeXt r=16",
            "Swin-Full FT", "Swin-LoRA r=8", "Hybrid r=16",
            "Hybrid r=8 (Proposed) ★"
        ],
        "Accuracy": ["54.66%", "87.73%", "73.10%", "93.60%", "95.57%", "97.66%", "98.02%", "98.37%"],
        "F1-Weighted": ["0.6201", "0.8854", "0.7580", "0.9386", "0.9575", "0.9770", "0.9805", "0.9839"],
        "AUC-ROC": ["0.6833", "0.9861", "0.9545", "0.9949", "0.9987", "0.9992", "0.9993", "0.9996"],
        "ECE ↓": ["0.1250", "0.0846", "0.1478", "0.0512", "0.0362", "0.0185", "0.0160", "0.0132"]
    }
    st.table(perf_data)

    c1, c2 = st.columns(2)
    with c1:
        st.image("imgs/modelcomp.jpg", caption="Accuracy Comparison Across All Configurations", use_container_width=True)
    with c2:
        st.image("imgs/roc.jpg", caption="ROC Curves — Macro AUC-ROC: 0.9996", use_container_width=True)

    st.write("##")

    # Training Dynamics
    st.markdown("#### Training Dynamics")
    c1, c2 = st.columns(2)
    with c1:
        st.image("imgs/valacc.jpg", caption="Training & Validation Accuracy (4-Fold CV)", use_container_width=True)
    with c2:
        st.image("imgs/valloss.jpg", caption="Training & Validation Loss (4-Fold CV)", use_container_width=True)

    st.write("##")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.image("imgs/confmat.jpg", caption="Confusion Matrix — Raw Counts & Normalised", use_container_width=True)

    st.write("##")

    # XAI
    st.markdown("#### Explainable AI (XAI) Visualisations")
    st.markdown("""
    Four complementary XAI methods are used for clinical interpretability:
    - **Grad-CAM:** Coarse localisation maps highlighting class-discriminative regions.
    - **Grad-CAM++:** Fine-grained multi-instance attribution for disjoint pathological regions.
    - **Integrated Gradients:** Axiomatic pixel-level importance scores.
    - **Attention Rollout:** Token-level attribution from the Swin Transformer branch.
    """)
    st.image("imgs/correctpred.jpg", caption="XAI Visualisations for Correctly Classified Samples", use_container_width=True)
    st.image("imgs/incorrect.jpg", caption="XAI Visualisations for Misclassified Samples", use_container_width=True)

    st.write("##")

    # Dataset
    st.markdown("#### Dataset: OASIS Brain MRI")
    dataset_data = {
        "CDR Rating": ["0", "0.5", "1", "2"],
        "Class": ["Non-Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"],
        "No. of Images": ["~67,200", "~13,700", "5,002", "488"]
    }
    st.table(dataset_data)
    st.markdown("Total: **86,437 images** across 4 severity classes from the OASIS Alzheimer's Detection dataset.")


elif selected_tab == 'How to Use':
    st.markdown("### About the Model")
    st.write("This application uses a **Hybrid ConvNeXt–Swin Transformer with Low-Rank Adaptation (LoRA r=8)** model, "
             "trained on the OASIS Brain MRI dataset (86,437 images).")
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", "98.37%")
        st.metric("AUC-ROC", "0.9996")
    with c2:
        st.metric("F1-Score (Weighted)", "0.9839")
        st.metric("ECE (Calibration)", "0.0132")
    with c3:
        st.metric("Trainable Parameters", "16.90M (29.77%)")
        st.metric("Cohen's Kappa", "0.9566")

    st.write("##")
    st.write("The model classifies brain MRI images into **4 stages** of Alzheimer's disease (by severity):")
    st.write("1. **Non Demented** — No signs of cognitive impairment")
    st.write("2. **Very Mild Demented** — Earliest detectable memory lapses")
    st.write("3. **Mild Demented** — Noticeable cognitive difficulties")
    st.write("4. **Moderate Demented** — Significant cognitive and functional decline")
    st.write("##")
    st.write("**How to use this tool:**")
    st.write("1. You need a soft copy of a brain MRI scan (PNG, JPEG, or JPG format).")
    st.write("2. Go to the **Alzheimer's Detection** tab and upload your MRI image.")
    st.write("3. Click **'Click to check'** — the model's prediction will appear in a blue dialog box below.")



elif selected_tab == 'About Team':
    st.write("##")

    st.markdown("### Research Team")
    st.markdown("*Institute of Information Technology, Jahangirnagar University, Dhaka-1342, Bangladesh*")
    st.write("##")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Farhana Chowdhury Ananda")
        pp = Image.open("profile pic/image.png")
        st.image(pp, output_format='PNG', width=250)
        st.markdown("""
        **Lead Researcher**

        Currently pursuing a Master's degree at the Institute of Information Technology, Jahangirnagar University.
        Completed BSc in Information Technology from the same institution.
        Previously worked as a Trainee Software Engineer at BJIT Ltd.
        Awarded the prestigious **National Science and Technology (NST) Fellowship** by the Ministry of Science and Technology.

        **Research Interests:** Deep Learning, Medical Image Processing, Computer Vision

        📧 farhana.c.ananda@gmail.com
        """)

    with col2:
        st.markdown("#### Dr. Jesmin Akhter")
        st.image("imgs/jesmin madam.jpg", width=250)
        st.markdown("""
        **Supervisor**

        Professor at the Institute of Information Technology, Jahangirnagar University, Savar, Dhaka-1342.
        Her research focuses on Internet of Things (IoT), Wireless Networks, Network Security, Software Engineering,
        Complexity of Algorithms, and Machine Learning.

        **Research Interests:** IoT, Wireless Networks, Network Security, Software Engineering, Machine Learning
        """)

    st.write("##")
    st.markdown("---")


