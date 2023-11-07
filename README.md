# Project LATACA: Local Automated Transcription and Content Analysis

## Overview
Project LATACA is designed to facilitate automatic transcription of interviews and to apply coding schemes for sentiment analysis. It features a front-end interface built with Streamlit and leverages machine learning models for speech recognition and content analysis.

## Installation

### Prerequisites:
- Python 3.9 or higher
- Conda package manager

### Setup Instructions:
1. **Fork the Repository**: Fork the project repository from GitHub to your account.

2. **Clone Your Fork**: Clone the repository to your local machine.
   ```git clone https://github.com/YOUR_USERNAME/QualresNLP.git``` or use the GitHub Desktop client.

3. **Conda Environment**:
   - Navigate to the project directory:
     ```
     cd QualresNLP
     ```
   - Create a conda environment using the `environment.yaml` file provided in the repository:
     ```
     conda env create -f environment.yaml
     ```
   - Activate the conda environment:
     ```
     conda activate qualresnlp
     ```

4. **Run the Streamlit App**:
   - Execute the following command:
     ```
     streamlit run app.py
     ```

## Models Used
- **Whisper**: An automatic speech recognition model for transcribing the interviews. We use a finetuned version that supports speaker diarization (recognising the differnt speakers in the audio) for 2 people. This model is based on the small.en model. 
- **BART**: A transformer model used for sequence classification tasks in the content analysis.

## Usage
- Launch the Streamlit app and select your interview audio file.
- The app will transcribe the audio and then apply the specified coding schemes.
- Adjust the number of sentences for analysis and confidence levels for classification as needed.
- Results can be exported in JSON or DOCX formats, and further analysis can be conducted with the provided content analysis tools.
- Note: Only english transcriptions are supprted presently.

## Features
- Transcription using Whisper model.
- Sentiment analysis and content coding using BART Large model.
- Data visualization with Seaborn and Plotly.
- Results export for further analysis in tools like NVivo.

## Contributing
- For feature requests, please open an issue in the [GitHub repository](https://github.com/JonasWeinert/QualresNLP/issues).
- To contribute to the project, fork the repo, make changes, and open a pull request for review.

## Contact
- For inquiries or assistance, reach out via [LinkedIn](https://www.linkedin.com/in/jweinert1997/), [GitHub](https://github.com/JonasWeinert/), or [Email](mailto:jonas.weinert@gmail.com).

## License
This codebase is free to use and modify for non-commercial use. Selling any of the technology used here to further parties requires explicit written consent.

**Note**: The accuracy of the transcription and content analysis may vary based on audio quality and the specificity of the coding schemes. It is recommended to review the transcriptions and analysis results for accuracy.

---

Made with 🧡 using Streamlit.

