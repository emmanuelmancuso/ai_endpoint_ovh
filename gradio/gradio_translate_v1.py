import gradio as gr
import requests
import base64
import json
import os
from typing import Dict, Any
from pydub import AudioSegment
import riva.client

#---------------------------------------------------------------------------------------------------------------------------------
# Choix des langues
language_choices = {
    "Anglais vers français": "en",
    "Français vers anglais": "fr"
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
def prepare_config(diarization: bool = False, word_time_offsets: bool = False) -> Dict[str, Any]:
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
        response.raise_for_status()  # Raise an exception for bad status codes
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
def translate_text(input_text: str, language_selection: str) -> str:
    """
    Translate the given text from the source language to the target language.
    
    Args:
        input_text (str): The text to be translated.
        source_lang (str): The source language of the text.
        target_lang (str): The target language to translate the text to.
    
    Returns:
        str: The translated text.
    
    Raises:
        Exception: If an error occurs during the translation process.
    """
    try:
        # Construire l'URI du service de traduction basé sur les langues source et cible
        uri = f"nvr-nmt-en-fr.endpoints-grpc.kepler.ai.cloud.ovh.net:443"
        
        # Configurer la connexion au service de traduction
        nmt_service = riva.client.NeuralMachineTranslationClient(
            riva.client.Auth(
                uri=uri,
                use_ssl=True,
            )
        )
        if language_selection == 'en':
            source_lang = 'en'
            target_lang = 'fr'
        elif language_selection == 'fr':
            source_lang = 'fr'
            target_lang = 'en'
        model_name = f'{source_lang}_{target_lang}_24x6'
        
        # Effectuer la traduction
        response = nmt_service.translate([input_text], model_name, source_lang, target_lang)
        
        return response.translations[0].text
    except Exception as e:
        return f"Erreur lors de la traduction : {str(e)}"
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
    }
    return languages.get(choice, "https://nvr-asr-fr-fr.endpoints.kepler.ai.cloud.ovh.net/api/v1/asr/recognize")
#---------------------------------------------------------------------------------------------------------------------------------
def transcribe_and_translate(audio_file, language_selection):
    """
    Transcribes the audio file and translate the transcription.

    Args:
        audio_file: The path to the audio file.
        source_lang: The language of the audio file.
        target_lang: The language to translate the transcription to.

    Returns:
        tuple: The transcription and the translation.

    Raises:
        Exception: If there is an error during the process.
    """
    try:
        api_key = os.environ.get("OVH_API_KEY", "")  # Utiliser une variable d'environnement pour la clé API
        processed_audio = convert_audio_to_16000hz(audio_file)
        audio_content = read_audio_file(processed_audio)
        url = choose_language(choice=language_selection)
        print(f"URL: {url}")
        config = prepare_config()
        transcription = transcribe_audio(url, api_key, audio_content, config)
        result = process_transcription(transcription)

        translation_result = translate_text(input_text=result, language_selection=language_selection)
        return result, translation_result
    except Exception as e:
        return f"An error occurred: {e}", ""
#---------------------------------------------------------------------------------------------------------------------------------
# Gradio Interface
def process_input(choice, audio_file=None, text_input=None, lang_selection='Anglais vers français'):
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
    if choice == "Audio":
        if audio_file is not None:
            return transcribe_and_translate(audio_file, language_selection=language_choices[lang_selection])
        return "Veuillez fournir une piste audio", ""
    elif choice == "Texte":
        translation_result = translate_text(input_text=text_input, language_selection=language_choices[lang_selection])
        return "", translation_result
    else:
        return "Choix invalide", ""
#---------------------------------------------------------------------------------------------------------------------------------
with gr.Blocks() as iface:
    choice_input = gr.Radio(choices=["Audio", "Texte"], label="Type d'entrée")
    
    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", visible=False, label="Source audio")
    
    text_input = gr.Textbox(lines=5, placeholder="Entrez le texte ici...", visible=False, label='Source texte')
    
    lang_selection = gr.Dropdown(choices=list(language_choices.keys()), label="Langue source", value="Anglais vers français")
    
    transcription_output = gr.Textbox(label="Transcription", visible=False)
    
    translation_output = gr.Textbox(label="Traduction", visible=False)

    def update_visibility(choice):
        if choice == "Audio":
            return gr.update(value=None, visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        elif choice == "Texte":
            return gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
    choice_input.change(fn=update_visibility, inputs=choice_input, outputs=[audio_input, text_input, lang_selection, transcription_output, translation_output])

    gr.Interface(
        fn=process_input,
        inputs=[
            choice_input,
            audio_input,
            text_input,
            lang_selection
        ],
        outputs=[
            transcription_output,
            translation_output
        ],
        title="""
            Traduction à partir d'un texte ou d'un enregistrement audio<br>
            Utilisation des <a href="https://endpoints.ai.cloud.ovh.net/" target="_blank">AI Endpoints d'OVHcloud</a><br><br>
            Codé par <a href="https://emmanuelmancuso.com/" target="_blank">Emmanuel Mancuso</a>👋🏼
            """,
        description="""
                    Téléchargez un fichier audio 🔊 ou entrez un texte 📃<br>
                    choisissez la langue source <br>
                    choisissez la langue cible <br>
                    Traduisez 🚀
                    """,
        allow_flagging='never'
    )
#---------------------------------------------------------------------------------------------------------------------------------
iface.launch(debug=True)