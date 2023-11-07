# Initialise project
import streamlit as st
import pandas as pd
import numpy as np

custom = 2


# Start Front End interface
st.set_page_config(page_title='LATACA', page_icon="🧡", layout="wide")




# Set Styles
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Supress streamlit branding
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

################### Front End ###################
# Set Front End appearance
st.title('Local Automated Transcription and Content Analysis') # Title
st.markdown('This app automatically transcribes interviews (ASR) & can apply coding schemes for sentiment analysis.') # First paragraph
st.subheader('Upload your Interview') # Upload prompt
st.markdown('Please slect your interview:') # First paragraph
uploaded_file = st.file_uploader('Choose your Interview') # Save file to memory for duration of session
# Sidebar 
with st.sidebar:
    st.title('Fancy a custom pipeline for your use case?')
    st.sidebar.write("")
    st.markdown('##### To get specialised advice and assistance on batch processing, custom workflows and other aspects of your data projects, reach out via: ')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/jweinert1997/)')
    st.markdown('[Github](https://github.com/JonasWeinert/)')
    st.markdown('[Email](emailto:jonas.weinert@gmail.com)')
    st.markdown('---')
    st.header('This project')
    st.markdown('- [Request a feature](https://github.com/JonasWeinert/QualresNLP/issues)')
    st.markdown('- [Contribute on GitHub](https://github.com/JonasWeinert/QualresNLP)')
    st.markdown('---')
    st.header('Feedback')
    st.markdown('---')
 

################# File Processing #################
# Save all four versions to JSON files
path_original = 'cleaned_transcript_original.json'
path_sentences = 'cleaned_transcript_sentences.json'
path_sentence_quads = 'cleaned_transcript_sentence_custom.json'

# Import Interview
import whisper
import os
import json

# Function to save uploaded file to disk and return the path
def save_uploadedfile(uploadedfile):
    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # This will create the temp directory if it doesn't exist
    temp_file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return temp_file_path

@st.cache_data  # The function below will be cached
def Automated_Speech_recognition(uploaded_file):
    # Save the uploaded file to disk
    saved_path = save_uploadedfile(uploaded_file)

    # Transcribe
    model = whisper.load_model("small.en-tdrz")
    result = whisper.transcribe(model, saved_path, verbose=True, condition_on_previous_text=True)

    # Save the result to 'result.json'
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    return result  # You may return the result or a path to the saved file

# Only call the processing function if the file is uploaded

if uploaded_file is not None:
    # Save the uploaded file to disk
    saved_path = save_uploadedfile(uploaded_file)

    # Transcribe
    model = whisper.load_model("small.en-tdrz")
    result = Automated_Speech_recognition(uploaded_file)
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
    if 'result.json' is not None:
        st.success('Transcription complete')
    st.markdown('----------')
st.subheader('Data Processing')
custom = st.number_input('Unit of analysis - enter number of sentences that will be used as the unit for classification by the ML model:', value=2, step=1)
# Get user input for labels
labels_input = st.text_input('Enter codes for the content analysis separated by commas', 'AI, safety, environment, gender')
# Split the input string into a list of labels
labels = [label.strip() for label in labels_input.split(',')]
conflevel = st.number_input('Confidence level - enter the minimum confidence level at which the ML model should assign a code to a segment. This usually takes some experimentation:', value=0.4, step=0.1, min_value=0.1, max_value=1.0)


################# Transcript Export #################
if 'result.json' is not None:
    tab1, tab2 = st.tabs(["Transcript", "QCA/Sentiment Analysis"])

# Clean and Export transcript
    import re
    # Load the JSON data
    with open('result.json', 'r') as file:
        transcription_result = json.load(file)

    # Remove 'before_speaker_turn' keys and split text at '[SPEAKER TURN]'
    cleaned_segments = []
    current_speaker = 1
    for segment in transcription_result['segments']:
        parts = segment['text'].split('[SPEAKER TURN]')
        for i, part in enumerate(parts):
            if part.strip():
                cleaned_segments.append({
                    'text': part.strip(),
                    'speaker': current_speaker,
                    'timestamp': segment['start'] if i == 0 else None
                })
            if i < len(parts) - 1:
                current_speaker = 2 if current_speaker == 1 else 1

    # Concatenate consecutive segments from the same speaker
    final_segments = []
    for segment in cleaned_segments:
        if not final_segments or segment['speaker'] != final_segments[-1]['speaker']:
            final_segments.append(segment)
        else:
            final_segments[-1]['text'] += ' ' + segment['text']

    # Function to split text into sentences
    def split_into_sentences(text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # Split each segment into sentences
    segments_sentences = []
    for segment in final_segments:
        sentences = split_into_sentences(segment['text'])
        for sentence in sentences:
            segments_sentences.append({
                'speaker': segment['speaker'],
                'text': sentence,
                'timestamp': segment['timestamp']
            })

    # Create groups of four sentences from the same speaker
    segments_sentence_quads = []
    quad = []

    for sentence in segments_sentences:
        if len(quad) > 0 and (sentence['speaker'] != quad[-1]['speaker'] or len(quad) == custom):
            # Once we have a quad, or the speaker changes, we append the current quad to the list
            segments_sentence_quads.append({
                'speaker': quad[0]['speaker'],
                'text': ' '.join([s['text'] for s in quad]),
                'timestamp': quad[0]['timestamp']
            })
            quad = []

        quad.append(sentence)

    # If there's a quad in progress when we finish, we add it as well
    if len(quad) > 0:
        segments_sentence_quads.append({
            'speaker': quad[0]['speaker'],
            'text': ' '.join([s['text'] for s in quad]),
            'timestamp': quad[0]['timestamp']
        })

    # Save all four versions to JSON files
    path_original = 'cleaned_transcript_original.json'
    path_sentences = 'cleaned_transcript_sentences.json'
    path_sentence_quads = 'cleaned_transcript_sentence_custom.json'

    with open(path_original, 'w') as file:
        json.dump(final_segments, file, ensure_ascii=False, indent=4)

    with open(path_sentences, 'w') as file:
        json.dump(segments_sentences, file, ensure_ascii=False, indent=4)


    with open(path_sentence_quads, 'w') as file:
        json.dump(segments_sentence_quads, file, ensure_ascii=False, indent=4)

    from datetime import timedelta

    # Helper function to format the timestamp
    def format_timestamp(seconds):
        # Converting seconds to hours, minutes, and seconds
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Formatting the timestamp. We exclude hours if it's 0.
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        else:
            return f"{minutes:02}:{seconds:02}"

    # Function to generate the transcript HTML entries
    def generate_transcript_html(transcript_data):
        transcript_html = ""
        for entry in transcript_data:
            speaker_class = f"speaker{entry['speaker']}"
            timestamp = format_timestamp(entry['timestamp'])
            transcript_html += f"""
            <div class="entry {speaker_class}">
                <span class="speaker-icon"></span>
                <span class="speaker">Speaker {entry['speaker']}</span>
                <p>{entry['text']}</p>
                <span class="timestamp">{timestamp}</span>
            </div>
            """
        return transcript_html

    # Load the JSON content from a file
    def load_json(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    # Main function to generate the HTML visualization
    def create_html_visualization(json_filepath, html_filepath):
        # Read the transcript data from the JSON file
        transcript_data = load_json(json_filepath)

        # Define the base HTML structure with inline CSS
        html_base = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interview Transcript</title>
        <style>
            /* Basic reset */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                padding: 20px;
            }

            .container {
                max-width: 800px;
                margin: auto;
                overflow: hidden;
                padding: 0 20px;
            }

            .transcript {
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                padding: 20px;
                margin-bottom: 20px;
            }

            .entry {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eaeaea;
            }

            .entry:last-child {
                border-bottom: none;
            }

            .speaker {
                font-weight: bold;
                margin-bottom: 5px;
            }

            .speaker-icon {
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 10px;
                vertical-align: middle;
            }

            .speaker1 .speaker-icon {
                background-color: #1e90ff;
            }

            .speaker2 .speaker-icon {
                background-color: #32cd32;
            }

            .timestamp {
                display: block;
                margin-top: 10px;
                color: #888;
                font-size: 0.8em;
            }

            /* Adding hover effect for entries */
            .entry:hover {
                background-color: #f8f8f8;
            }
        </style>
        </head>
        <body>
        <div class="container">
            <div class="transcript">
                <!-- Transcript entries will be inserted here -->
            </div>
        </div>
        </body>
        </html>
        """

        # Generate the HTML content for the transcript
        transcript_html_content = generate_transcript_html(transcript_data)

        # Insert the transcript HTML content into the base HTML structure
        html_content = html_base.replace('<!-- Transcript entries will be inserted here -->', transcript_html_content)

        # Save the HTML content to a file
        with open(html_filepath, 'w') as file:
            file.write(html_content)


    json_file_path = 'cleaned_transcript_original.json' 
    output_html_file_path = 'transcript_visualization.html'  # The output HTML file path
    create_html_visualization(json_file_path, output_html_file_path)


    # Word
    from docx import Document
    from bs4 import BeautifulSoup

    def create_word_from_html_file(html_file_path, word_file_path):
        document = Document()
        
        # Open the HTML file and read its content
        with open(html_file_path, 'r') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')

        for entry in soup.find_all('div', class_='entry'):
            speaker = entry.find('span', class_='speaker').text
            text = entry.find('p').get_text()
            timestamp = entry.find('span', class_='timestamp').text

            document.add_paragraph(f"{speaker}: {text} ({timestamp})")

        document.save(word_file_path)

    #Export
    html_file_path = 'transcript_visualization.html'  # Path to your HTML file
    word_file_path = 'transcript.docx'
    create_word_from_html_file(html_file_path, word_file_path)


with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('##### Preview')
                # Read the HTML content from a file
        with open(output_html_file_path, 'r') as file:
            html_content = file.read()
        import streamlit.components.v1 as  components
        st.components.v1.html(html_content, width=None, height=7000, scrolling=True)
    with col2:
        st.markdown('##### Export')
        # For the original cleaned transcript JSON
        with open(path_original, 'r') as f:
            st.download_button(
                label="Save Original Transcript (JSON)",
                data=f,
                file_name="cleaned_transcript_original.json",
                mime="application/json"
            )

        # For the sentences cleaned transcript JSON
        with open(path_sentences, 'r') as f:
            st.download_button(
                label="Save sentence level Transcript (JSON)",
                data=f,
                file_name="cleaned_transcript_sentences.json",
                mime="application/json"
            )

        # For the custom sentence quads cleaned transcript JSON
        with open(path_sentence_quads, 'r') as f:
            st.download_button(
                label="Save Custom UoA Transcript (JSON)",
                data=f,
                file_name="cleaned_transcript_sentence_custom.json",
                mime="application/json"
            )

        # For the Word document
        with open(word_file_path, 'rb') as f:
            st.download_button(
                label="Save Original Transcript (DOCX)",
                data=f,
                file_name="transcript.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        

    with tab2:
        # Classification
        import json
        from transformers import BartForSequenceClassification, BartTokenizer, pipeline


        with open('cleaned_transcript_sentence_custom.json', 'r') as file:
            data = json.load(file)


        model_name = 'facebook/bart-large-mnli'
        model = BartForSequenceClassification.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        #labels = ['AI', 'safety', 'environment', 'gender', 'innovation', 'politics', 'positive sentiment', 'negative sentiment', 'excitement', 'worry']


        classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)

        @st.cache_data  # The function below will be cached
        def Content_Analysis(text, labels):
            return classifier(text, candidate_labels=labels, multi_label=True)

        results = [Content_Analysis(entry['text'], labels) for entry in data]

        for entry, result in zip(data, results):
            entry['labels'] = result['labels']
            entry['scores'] = result['scores']

        with open('classified_transcript.json', 'w') as outfile:
            json.dump(data, outfile)
            st.write('')

        # Export & Display
        # Export classification as tables
        import pandas as pd

        filtered_flattened_data = []
        @st.cache_data  # The function below will be cached
        def process_classification(data, results, threshold=conflevel):
            filtered_flattened_data = []
            # Go through the original data and corresponding classification results
            for entry, result in zip(data, results):
                for label, score in zip(result['labels'], result['scores']):
                    # Only include labels with a score higher than the threshold
                    if score > threshold:
                        # Create a new record for each label with a score above the threshold
                        filtered_flattened_data.append({
                            'text': entry['text'],
                            'label': label,
                            'score': score,
                            'timestamp': entry['timestamp'],
                            'speaker': entry['speaker']
                        })
            return filtered_flattened_data

        # Assuming `data` and `results` are defined elsewhere in your app
        filtered_flattened_data = process_classification(data, results)

        # Convert the filtered flattened data to a DataFrame
        df = pd.DataFrame(filtered_flattened_data)
        df = df[df['score'] >= conflevel]


        # Export the DataFrame to a CSV file
        csv_file_path = 'long_classified_transcript.csv'
        df.to_csv(csv_file_path, index=False)

        # Wide
        # Group by 'text' and aggregate the 'label' and 'score' in the required format
        grouped_df = df.groupby(['timestamp', 'speaker', 'text']).apply(
            lambda x: ', '.join([f"{row['label']} ({row['score']:.2f})" for index, row in x.iterrows()])
        ).reset_index()

        # Rename the aggregated column as 'label'
        grouped_df.columns = ['timestamp', 'speaker', 'text', 'label']


        grouped_df.to_csv("wide_classified_transcript.csv", index=False)

        #Nvivo
        from datetime import datetime, timedelta

        base_datetime = datetime(2023, 1, 1, 0, 0, 0)

        # Create a new column with standardized datetime format
        df['timestamp'] = df['timestamp'].apply(lambda x: (base_datetime + timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M:%S'))

        # Save the modified dataframe to a new CSV file for import into NVivo
        nvivo_ready_csv_path = 'nvivo_ready_classified_transcript.csv'
        df.to_csv(nvivo_ready_csv_path, index=False)

        with st.expander("Edit Codes"):
            st.markdown('Score: Confidence level at which the ML model assigned the code to the segment')
            heatmapdf = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        
        # Correlation matrix heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np


        # Filter out rows with scores lower than 0.35
        df_filtered = heatmapdf

        # Display the first few rows of the filtered dataframe
        #print(df_filtered.head())

        # Convert the DataFrame to a one-hot encoded matrix
        one_hot_encoded = pd.get_dummies(df_filtered['label']).groupby(df_filtered['text']).max()

        # Compute the co-occurrence matrix
        co_occurrence_matrix = one_hot_encoded.T.dot(one_hot_encoded)

        # Re-calculate one-hot encoded matrix
        one_hot_encoded = pd.get_dummies(df_filtered['label'])

        # Add the text as an index to group by
        one_hot_encoded['text'] = df_filtered['text']

        # Group by text and sum to get the frequency of each label per text
        label_frequency_per_text = one_hot_encoded.groupby('text').sum()

        # Calculate the co-occurrence matrix again, this time it should be the frequency of co-occurrences
        co_occurrence_frequency_matrix = label_frequency_per_text.T.dot(label_frequency_per_text)

        # Since the matrix will be symmetric and have a diagonal of the sum of each label's occurrences,
        # we can zero out the diagonal for a cleaner look in the heatmap.
        np.fill_diagonal(co_occurrence_frequency_matrix.values, 0)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(co_occurrence_matrix, dtype=int))

        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))

        # Draw the heatmap with the mask
        sns.heatmap(co_occurrence_frequency_matrix, mask=mask, cmap='viridis', vmax=None, square=True, annot=True, fmt='d')

        # Adjust the layout
        plt.title('Label Co-occurrence Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        #Interactive option
        import plotly.figure_factory as ff

        fig = ff.create_annotated_heatmap(
            z=co_occurrence_frequency_matrix.values,
            x=co_occurrence_frequency_matrix.columns.tolist(),
            y=co_occurrence_frequency_matrix.index.tolist(),
            annotation_text=co_occurrence_frequency_matrix.values.astype(str),
            showscale=True,
            colorscale='Viridis'
        )

        fig.update_layout(
            title='Co-occurrence Frequencies',
            xaxis=dict(tickangle=-45, side='top'),
            yaxis=dict(tickmode='array', tickvals=np.arange(len(co_occurrence_frequency_matrix.index))),
            margin=dict(t=90, l=120)
        )

        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 8

        column1, column2 = st.columns(2)


        # Visualize codes
        # Classified transcript as html
        import json

        # Function to format the timestamp
        def format_timestamp(seconds):
            # Assuming the timestamp is in seconds and you want to convert it to hh:mm:ss
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        # Function to generate HTML content for the transcript with tooltips for labels and scores,
        # dropping labels with scores less than 0.4
        def generate_transcript_html_with_filtered_tooltips(transcript_data):
            transcript_html = ""
            for entry in transcript_data:
                speaker_class = f"speaker{entry['speaker']}"
                timestamp = format_timestamp(entry['timestamp'])
                
                # Filter out labels with scores less than 0.4
                filtered_labels_scores = [
                    (label, score) for label, score in zip(entry['labels'], entry['scores']) if score >= conflevel
                ]
                
                # Prepare the labels and scores in a tooltip format if there are any after filtering
                if filtered_labels_scores:
                    labels_scores = ", ".join(
                        f"{label} ({score:.2f})" for label, score in filtered_labels_scores
                    )
                    tooltip = f" data-tooltip='{labels_scores}'"
                else:
                    tooltip = ""

                transcript_html += f"""
                <div class="entry {speaker_class}"{tooltip}>
                    <span class="speaker-icon"></span>
                    <span class="speaker">Speaker {entry['speaker']}</span>
                    <p>{entry['text']}</p>
                    <span class="timestamp">{timestamp}</span>
                </div>
                """
            return transcript_html

        # Load the JSON content from the uploaded file
        def load_transcript_data(json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            return data

        # HTML and CSS template
        html_base = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transcript Visualization</title>
        <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .entry { margin-bottom: 10px; padding: 10px; border-left: 3px solid #808080; }
        .speaker-icon { display: inline-block; width: 20px; height: 20px; background-color: #4CAF50; border-radius: 50%; margin-right: 10px; }
        .speaker2 .speaker-icon { background-color: #2196F3; }
        .speaker { font-weight: bold; margin-right: 5px; }
        .timestamp { font-size: 0.85em; color: #555; }
        [data-tooltip] {
        position: relative;
        cursor: pointer;
        overflow: visible;
        }
        [data-tooltip]:before {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        white-space: pre-wrap;
        max-width: 200px;
        padding: 2px;
        border-radius: 5px;
        background: rgba(0, 0, 0, 0.7);
        color: #fff;
        font-size: 12px;
        text-align: left;
        visibility: hidden;
        opacity: 0;
        transition: visibility 0s, opacity 0.5s ease;
        z-index: 100;
        }
        [data-tooltip]:hover:before, [data-tooltip]:hover:after {
        visibility: visible;
        opacity: 1;
        }
        [data-tooltip]:after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0, 0, 0, 0.7) transparent transparent transparent;
        visibility: hidden;
        opacity: 0;
        transition: visibility 0s, opacity 0.5s ease;
        }
        </style>
        </head>
        <body>
        <!-- Transcript entries will be inserted here -->
        </body>
        </html>
        """

        # Main script to load data, generate HTML, and save it to a file
        def main(json_file_path, output_html_file_path):
            # Load the transcript data from the JSON file
            transcript_data = load_transcript_data(json_file_path)
            
            # Generate the full HTML content with filtered tooltips
            full_transcript_html = generate_transcript_html_with_filtered_tooltips(transcript_data)
            
            # Insert the full transcript HTML content into the base HTML structure
            full_html_content = html_base.replace('<!-- Transcript entries will be inserted here -->', full_transcript_html)
            
            # Save the full HTML content to a file
            with open(output_html_file_path, 'w') as file:
                file.write(full_html_content)
            return full_html_content

        # Use the main function with the path to the JSON file and the desired output HTML file path
        json_file_path = 'classified_transcript.json'  # Replace with the path to your JSON file
        output_html_file_path = 'transcript_with_tooltips.html'
        full_html_content = main(json_file_path, output_html_file_path)


        with column1:
            # Show the plot
            #st.pyplot(plt)
            st.plotly_chart(fig, use_container_width=True)
            def get_csv_download_link(data):
                # Convert the processed data into a DataFrame
                df = pd.DataFrame(data)
                # Convert DataFrame to CSV
                csv = df.to_csv(index=False)
                # Convert CSV to bytes
                b_csv = csv.encode('utf-8')
                return b_csv

            # Call the function to get the CSV bytes
            csv_bytes = get_csv_download_link(filtered_flattened_data)

            # Create a download button
            st.download_button(
                label="Download coded dataset as CSV",
                data=csv_bytes,
                file_name="classification_results.csv",
                mime="text/csv",
            )
            st.info("Which data format would you prefer this in? Let me know in the feedback section!")

        with column2:
            st.write('Hover over the cells to see the labels and confidence scores')
            st.components.v1.html(full_html_content, width=None, height=7000, scrolling=True)
        






################# Footer #################
st.markdown('---')
with st.expander("Privacy and legal note"):
    st.markdown('Functional Cookies: ODK Cleaning Code Generator uses functional cookies to enhance your user experience on our webapp. These cookies are essential for the basic functionality of the webapp, such as remembering your preferences, providing security, and improving site performance. No personal data is collected. Please note that by using this webapp, you agree to this use of functional cookies. You can, however, disable cookies through your browser settings, but this may affect the functionality of the webapp. Legal Note: ODK Cleaning Code Generator is provided "as is" without any representations or warranties, express or implied. ODK Cleaning Code Generator makes no representations or warranties in relation to the information, services, or materials provided on our webapp. ODK Cleaning Code Generator does not accept liability for any inaccuracies, errors, or omissions in the information, services, or materials provided on our webapp. By using this webapp, you acknowledge that the information and services may contain inaccuracies or errors, and ODK Cleaning Code Generator expressly excludes liability for any such inaccuracies or errors to the fullest extent permitted by law. ODK Cleaning Code Generator is not responsible or liable for any outcomes or consequences resulting from the use of the webapp or any of its features. You agree that your use of the webapp is at your sole risk, and you assume full responsibility for any decisions or actions taken based on the information or materials provided. By using ODK Cleaning Code Generator, you agree to indemnify, defend, and hold harmless ODK Cleaning Code Generator and its creator from and against any and all claims, liabilities, damages, losses, or expenses, including reasonable attorneys fees and costs, arising out of or in any way connected with your access to or use of the webapp.')



    
