# Medibeng-Orpheus-3b-0.1-ft Fine-Tuning Process
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Model Size](https://img.shields.io/badge/size-3B-ff69b4)
![Base Model](https://img.shields.io/badge/Base_Model-Llama--3.2--3B--Instruct-blue)
[![Model on HuggingFace](https://img.shields.io/badge/HuggingFace-medibeng--orpheus--3b--0.1--ft-orange?logo=huggingface&logoColor=yellow)](https://huggingface.co/The-Data-Dilemma/Medibeng-Orpheus-3b-0.1-ft)



**Medibeng-Orpheus-3b-0.1-ft** is a fine-tuned Text-to-Speech (TTS) model trained on the **MediBeng** dataset, designed to handle bilingual Bengali-English **code-switching** in healthcare settings. This model builds on the **Orpheus TTS** architecture and leverages the **LLaMA-3b** backbone. Special thanks to **Unsloth** for their contribution in accelerating the training process using **HuggingFace's TRL** library.

### **Demo Video**

Watch the demo video showcasing how the model performs with Bengali-English code-switching in healthcare dialogues. The video demonstrates its ability to generate realistic TTS output in real-world scenarios.

* Watch the **Demo Video** here:
  

https://github.com/user-attachments/assets/94f39f5b-30c9-49bd-b195-69e1dced890c


  

## **🚀 Model Overview**

The **Medibeng-Orpheus-3b-0.1-ftt** model produces high-quality speech from text with support for Bengali-English code-switching, specifically designed for healthcare-related applications. It simulates patient-doctor interactions in a bilingual context, demonstrating state-of-the-art performance in clinical dialogue generation.

### **Key Features:**

* **Bilingual Code-Switching:** Handles seamless transitions between **Bengali** and **English** in a conversational setting.
* **Healthcare Focus:** Tailored for use in healthcare, simulating medical dialogues between patients and doctors.
* **Accelerated Fine-Tuning:** Using **Unsloth** and **HuggingFace’s TRL**, the fine-tuning process was optimized for **2x faster** training.

## **📊 Fine-Tuning Process**

Here’s a quick overview of the fine-tuning process that was followed to train the model:

| **Step**                   | **Description**                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Dataset Preparation** | Used **MediBeng**, a bilingual Bengali-English healthcare dataset, which simulates clinical dialogues.                         |
| **2. Model Architecture**  | Built on **Orpheus TTS** using **LLaMA-3b** architecture for high-quality speech synthesis.                                    |
| **3. Fine-Tuning**         | Fine-tuned using **HuggingFace’s TRL** library, supported by **Unsloth** for accelerated training.                             |
| **4. Training Duration**   | **2x faster** fine-tuning thanks to optimization techniques from **Unsloth**.                                                  |
| **5. Validation**          | Ensured the model’s ability to handle code-switching, and clinical dialogues, achieving satisfactory results in initial tests. |

## **⚙️ Usage**

### **Colab Notebook Interface**

To easily interact with the fine-tuned model, use the provided **Colab Notebook**. The notebook includes all necessary steps to run the model, generate speech, and experiment with different inputs.

* Access the **Colab Notebook Interface** here:
* 
  - 🔗 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg) Fine-Tuning Colab Notebook by Unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)
  - ⚡ [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg) Quick Access Medibeng-Orpheus-3b-0.1-ft ](https://colab.research.google.com/drive/1J5i_pTV4BmDxXlmyK1ZwESrqiIpy4oJo?usp=sharing)
  - 🤗 [Model on HuggingFace](https://huggingface.co/The-Data-Dilemma/Medibeng-Orpheus-3b-0.1-ft)


## **📋 Limitations**

* **Pronunciation & Prosody:** Although the model generates high-quality speech, further fine-tuning is needed to improve pronunciation nuances and prosody, especially for diverse accents.
* **Accent Handling:** The model is focused on general code-switching and may need further tuning for region-specific accents.
* **Dataset Bias:** The **MediBeng** dataset is a limited representation of healthcare dialogues and may not cover every healthcare context or dialect.

## **📝 Use Cases**

This model is suitable for various healthcare-related applications, including:

* **Virtual Healthcare Assistants:** Bilingual assistants for seamless patient interaction.
* **Telemedicine Applications:** Assisting doctors and patients in remote consultations with bilingual dialogue generation.
* **Medical Training Simulations:** Simulating bilingual clinical interactions for medical education.
* **Bilingual Healthcare Chatbots:** Enhancing conversational agents with natural speech synthesis in both Bengali and English.

## **📂 Dataset**

The model was fine-tuned on the **MediBeng** dataset:

* **Content:** Contains code-switched Bengali-English synthetic dialogues, simulating real-world patient-doctor conversations.
* **Scope:** Designed to cover a variety of clinical interactions in bilingual settings.

For more details on the dataset, visit the **MediBeng Dataset** documentation [here](https://huggingface.co/datasets/pr0mila-gh0sh/MediBeng).


## **📑 License**

This model is provided under the **Apache 2.0 License**. Feel free to use, modify, and distribute the model as long as you comply with the terms of the license.

