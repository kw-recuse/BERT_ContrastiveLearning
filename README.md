# Korean Resume-JD Matching Model

## Dataset

Created synthetic datasets with 29,800 positive/negative pairs each using the GPT-4o-mini API. The discrepancy between Dataset-V1 and Dataset-V2 lies in how the preprocessing step was applied to the resume sequence. The former contains the full resume, while the latter includes only the experience, education, and skill sections.

- [Dataset-V1](https://huggingface.co/datasets/recuse/resume-jd-match-kr)
- [Dataser-V2](https://huggingface.co/datasets/recuse/datasetV2)

## Ablation Study Result

Conducted ablation studies varying two approaches: cross-encoder and single-encoder, fine-tuning multilingual BERT models with the custom dataset using contrastive loss.

<table>
  <tr>
    <th align="center">Single Encoder</th>
    <th align="center">Cross Encoder</th>
  </tr>
  <tr>
    <td align="center" style="vertical-align:middle;">
      <img src="https://github.com/user-attachments/assets/ef137a59-48ea-4309-85d3-cde0c4165b63" alt="Single Encoder" style="max-width:100%; height:auto;"/>
    </td>
    <td align="center" style="vertical-align:middle;">
      <img src="https://github.com/user-attachments/assets/67c2e592-6cbb-48da-8543-cabb8a8cc028" alt="Cross Encoder" style="max-height:50%; width:auto;"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      The similarity score between the resume and the job description is calculated by computing the cosine similarity between their embeddings, obtained by passing the raw sequences through the BERT model.
    </td>
    <td align="center">
      The resume and job description strings are concatenated and then passed through the BERT model. All the embeddings are fed into an MLP layer, which produces a single similarity score.
    </td>
  </tr>
</table>



### Single Encoder

| Model                      | dataV1  | dataV2  | # Params |
|----------------------------|---------|---------|----------|
| TD-IDF                     | 0.2214  | 0.2267  | NA       |
| OpenAI-text-embedding-large-3 | ...     | 0.2853  | Unknown  |
| mBERT                      | 0.3887  | 0.3449  | 135M     |
| mPnet                      | 0.3936  | 0.3254  | 278M     |
| Bge-m3-korean              | 0.3621  | 0.3394  | 568M     |
| **Below are ours**         |         |         |          |
| mBERT+MLM                  | 0.4612  | 0.4250  | 135M     |
| mBERT + CLoss              | 0.1066  | 0.1056  | 135M     |
| mBERT+MLM+CLoss            | 0.1084  | ...     | 135M     |
| mPnet + CLoss              | **0.1038** | **0.1024** | 278M     |
| Bge-m3-korean + CLoss      | 0.1085  | 0.1052  | 568M     |


#### Cross Encoder

| Model                       | MSE(dataV1) | MSE(dataV2) | # Params |
|-----------------------------|-------------|-------------|----------|
| Bge-m3-korean + MLP + BCE   | 0.0016      | 0.0803      | 568M     |


## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kw-recuse/BERT_ContrastiveLearning.git](https://github.com/kw-recuse/BERT_ContrastiveLearning.git)
    ```

2.  **Navigate to the directory:**
    ```bash
    cd BERT_ContrastiveLearning
    ```

3.  **Run the training:**

    You can start the training process by importing and using the `Trainer` class, as shown in the example below (e.g., in a Python script or notebook).

    ```python
    import sys
    
    # Add the repository path if needed (e.g., if running from a notebook outside the main directory)
    # sys.path.append('/path/to/BERT_ContrastiveLearning')

    from scripts.train import Trainer

    # Initialize the Trainer with your configuration
    trainer = Trainer(
        config_file="configs/train/multiling_BERT.json",
        checkpoints_path="checkpoints", 
        csv_file_path="output_file.csv", # path to downloaded csv file from Huggingface
        col_name1='resume',
        col_name2='jd',
        label_col='label'
    )

    # Start training
    trainer.train()
    ```
## Future Plans

- Knowledge Distillation
