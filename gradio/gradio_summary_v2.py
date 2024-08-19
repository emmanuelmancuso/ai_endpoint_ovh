import gradio as gr
import requests
import base64
import json
import os
from typing import Dict, Any
from pydub import AudioSegment

# Dictionnaire de couleurs ANSI pour différents interlocuteurs
colors = {
    1: "\033[91m",  # Rouge
    2: "\033[92m",  # Vert
    3: "\033[94m",  # Bleu
    4: "\033[93m",  # Jaune
    'reset': "\033[0m"  # Réinitialiser la couleur
}

#---------------------------------------------------------------------------------------------------------------------------------
def read_audio_file(file_path: str) -> bytes:
    """
    Reads an audio file from the specified file path.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        bytes: The content of the audio file.

    Raises:
        IOError: If there is an error reading the audio file.
    """
    try:
        with open(file_path, "rb") as audio_file:
            return audio_file.read()
    except IOError as e:
        raise IOError(f"Erreur de lecture du fichier audio : {e}")
#---------------------------------------------------------------------------------------------------------------------------------
def prepare_config(diarization: bool = True, word_time_offsets: bool = False) -> Dict[str, Any]:
    """
    Prepares the configuration for the audio transcription.

    Args:
        diarization (bool): Whether to add speaker diarization to the configuration.
        word_time_offsets (bool): Whether to enable word time offsets in the configuration.

    Returns:
        Dict[str, Any]: The prepared configuration.
    """
    return {
        "add_speaker_diarization_to_config": diarization,
        "enable_word_time_offsets": word_time_offsets,
        "add_word_boosting_to_config": None,
        "enable_automatic_punctuation":True,
    }
#---------------------------------------------------------------------------------------------------------------------------------
def transcribe_audio(url: str, api_key: str, audio_content: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends a request to the API to transcribe the audio content.

    Args:
        url (str): The API endpoint URL.
        api_key (str): The API key for authentication.
        audio_content (bytes): The audio content to be transcribed.
        config (Dict[str, Any]): The configuration for the transcription.

    Returns:
        Dict[str, Any]: The transcription result.

    Raises:
        Exception: If there is an error during the API request.
    """
    headers = {"X-Ovh-Application": api_key}
    
    config_json = json.dumps(config)
    
    files = {
        'audio': ('audio.wav', audio_content, 'audio/wav'),
        'config': ('config.json', config_json, 'application/json')
    }
    
    try:
        response = requests.post(url, files=files, headers=headers)
        print(f"{response.json() =}")
        
        for entry in response.json():
            words = entry['alternatives'][0]['words']
            dialogue = ""
            last_speaker = None
            
            for word_info in words:
                word = word_info['word']
                speaker = word_info['speaker_tag']
                
                # Si le locuteur change, imprimer la ligne précédente
                if speaker != last_speaker and last_speaker is not None:
                    print(f"{colors[last_speaker]}{dialogue.strip()}{colors['reset']}")
                    dialogue = ""
                
                dialogue += word + " "
                last_speaker = speaker
            
            # Imprimer la dernière partie du dialogue
            if dialogue:
                print(f"{colors[last_speaker]}{dialogue.strip()}{colors['reset']}")

        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erreur lors de la requête à l'API : {e}")
#---------------------------------------------------------------------------------------------------------------------------------
def process_transcription(transcription: Dict[str, Any]) -> str:
    """
    Processes the transcription result and returns a concatenated transcript.

    Args:
        transcription (Dict[str, Any]): The transcription result.

    Returns:
        str: The concatenated transcript.
    """
    full_transcript = ""
    for sentence in transcription:
        for alternative in sentence["alternatives"]:
            full_transcript += alternative["transcript"] + " "
    return full_transcript.strip()
#---------------------------------------------------------------------------------------------------------------------------------
def summary(content: str, context: str, choice: str) -> str:
    """
    Generates a summary for the given content using the specified language model.

    Args:
        content (str): The content to be summarized.
        choice (str): The language model choice.

    Returns:
        str: The generated summary.

    Raises:
        Exception: If there is an error during the API request.
    """
    if choice == "Llama-3-70B-Instruct":
        url = "https://llama-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Meta-Llama-3-70B-Instruct"
    elif choice == "Llama-3-8B-Instruct":
        url = "https://Llama-3-8B-Instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Meta-Llama-3-8B-Instruct"
    elif choice == "Mistral-7B-Instruct":
        url = "https://Mistral-7B-Instruct-v02.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Mistral-7B-Instruct-v0.2"
    elif choice == "Mixtral-8x22b-Instruct":
        url = "https://Mixtral-8x22B-Instruct-v01.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Mixtral-8x22B-Instruct-v0.1"
    elif choice == "Mixtral-8x7b-instruct":
        url = "https://mixtral-8x7b-instruct-v01.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Mixtral-8x7B-Instruct-v0.1"
    else:
        url = "https://llama-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
        model="Meta-Llama-3-70B-Instruct"
    
    prompt = f"""
                Vous êtes mon assistant et votre seule mission est de me fournir un résumé du texte situé entre <>.
                Le texte supplémentaire présent entre ## (s'il n'est pas vide) fournit des éléments de contexte à ce texte.
                Veuillez fournir votre réponse en français uniquement, avec une longueur maximale de 5 phrases.
                Texte : <{content}>
                Contexte : #{context}#
            """
    
    payload = {
        "max_tokens": 512,
        "messages": [
            {
                "content":  prompt,
                "name": "User",
                "role": "user"
            }
        ],
        "model": model,
        "temperature": 0,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OVH_API_KEY')}",
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        choices = response_data["choices"]
        for choice in choices:
            return choice["message"]["content"]
    else:
        return f"Error: {response.status_code}"
#---------------------------------------------------------------------------------------------------------------------------------
def choose_language(choice: str) -> str:
    """
    Returns the appropriate API URL based on the selected language.

    Args:
        choice (str): The language choice.

    Returns:
        str: The API URL for the selected language.
    """
    languages = {
        "fr": "https://nvr-asr-fr-fr.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize",
        "en": "https://nvr-asr-en-gb.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize",
        "en/us": "https://nvr-asr-en-us.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize",
        "de": "https://nvr-asr-de-de.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize",
        "es": "https://nvr-asr-es-es.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize",
        "it": "https://nvr-asr-it-it.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize"
    }
    return languages.get(choice, "https://nvr-asr-fr-fr.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize")
#---------------------------------------------------------------------------------------------------------------------------------
def convert_audio_to_16000hz(input_file: str) -> str:
    """
    Converts the given audio file to 16000 Hz and returns the path to the converted file.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        str: The path to the converted audio file.

    Raises:
        Exception: If there is an error during the conversion.
    """
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        output_file = "data/processed_audio.wav"
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion du fichier audio : {e}")
#---------------------------------------------------------------------------------------------------------------------------------
def transcribe_and_summarize(audio_file, language_choice, llm_choice, context):
    """
    Transcribes the audio file and summarizes the transcription using the specified language model.

    Args:
        audio_file: The path to the audio file.
        language_choice: The selected language for transcription.
        llm_choice: The selected language model for summarization.

    Returns:
        tuple: The transcription and the summary.

    Raises:
        Exception: If there is an error during the process.
    """
    try:
        api_key = os.environ.get("OVH_API_KEY", "")  # Utiliser une variable d'environnement pour la clé API
        processed_audio = convert_audio_to_16000hz(audio_file)
        audio_content = read_audio_file(processed_audio)
        url = choose_language(language_choice)        
        config = prepare_config()
        transcription = transcribe_audio(url, api_key, audio_content, config)
        result = process_transcription(transcription)
        
        summary_result = summary(content=result, context=context, choice=llm_choice)
        return result, summary_result[:-10]
    except Exception as e:
        return f"Une erreur est survenue : {e}", ""
#---------------------------------------------------------------------------------------------------------------------------------
# Gradio Interface
def process_input(choice, audio_file=None, text_input=None, lang_choice=None, llm_choice=None, context_input=None):
    """
    Processes the input based on the user's choice (Audio or Text).

    Args:
        choice: The type of input (Audio or Text).
        audio_file: The audio file input.
        text_input: The text input.
        lang_choice: The language choice for audio transcription.
        llm_choice: The language model choice for summarization.

    Returns:
        tuple: The transcription (if audio) and the summary.
    """
    max_chars = 200
    if len(context_input) > max_chars:
        return "", "Le contexte ne peut pas excéder 200 caractères."
    else :
        if choice == "Audio":
            if audio_file is not None:
                return transcribe_and_summarize(audio_file, lang_choice, llm_choice, context=context_input,)
            return "Veuillez fournir une piste audio", ""
        elif choice == "Texte":
            summary_result = summary(content=text_input, context=context_input, choice=llm_choice)
            return "", summary_result[:-10]
        else:
            return "Choix invalide", ""
#---------------------------------------------------------------------------------------------------------------------------------
language_choices = ["fr", "en", "en/us", "de", "es", "it"]
llm_choices = ["Llama-3-70B-Instruct", "Llama-3-8B-Instruct", "Mistral-7B-Instruct", "Mixtral-8x22b-Instruct", "Mixtral-8x7b-instruct"]
#---------------------------------------------------------------------------------------------------------------------------------
with gr.Blocks() as iface:
    choice_input = gr.Radio(choices=["Audio", "Texte"], label="Type d'entrée")
    
    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", visible=False, label="Source audio")
    
    text_input = gr.Textbox(lines=5, placeholder="Entrez le texte ici...", visible=False, label='Source texte')
    
    lang_dropdown = gr.Dropdown(value=language_choices[0], choices=language_choices, label="Choisissez la langue de l'audio", visible=False)
    
    llm_dropdown = gr.Dropdown(value=llm_choices[0], choices=llm_choices, label="Choisissez le LLM pour effectuer le résumé", visible=False)
    
    context_input = gr.Textbox(lines=8, placeholder="Entrez le contexte ici (optionnel).\n\nExemples :\nIl s'agit d'un appel téléphonique professionnel entre...\nIl s'agit d'une émission de radio qui parle de...\nCe texte scientifique parle de...\n\nTaille du contexte limitée à 200 caractères", visible=False, label="Contexte du contenu")
    
    transcription_output = gr.Textbox(label="Transcription", visible=False)
    
    summary_output = gr.Textbox(label="Résumé", visible=False)

    def update_visibility(choice):
        if choice == "Audio":
            return gr.update(value=None, visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value="", visible=True)
        elif choice == "Texte":
            return gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    choice_input.change(fn=update_visibility, inputs=choice_input, outputs=[audio_input, text_input, lang_dropdown, llm_dropdown, context_input, transcription_output, summary_output])

    gr.Interface(
        fn=process_input,
        inputs=[
            choice_input,
            audio_input,
            text_input,
            lang_dropdown,
            llm_dropdown,
            context_input
        ],
        outputs=[
            transcription_output,
            summary_output
        ],
        title="""
            Créer un résumé à partir d'un texte ou d'un enregistrement audio<br>
            Utilisation des <a href="https://endpoints.ai.cloud.ovh.net/" target="_blank">AI Endpoints d'OVHcloud</a><br><br>
            Codé par <a href="https://emmanuelmancuso.com/" target="_blank">Emmanuel Mancuso</a>👋🏼
            """,
        description="""
                    Téléchargez un fichier audio 🔊 ou entrez un texte 📃<br>
                    choisissez la langue pour l'audio <br>
                    choisissez le modèle de LLM pour obtenir le résumé 🚀
                    """,
        allow_flagging='never'
    )
#---------------------------------------------------------------------------------------------------------------------------------
iface.launch()
