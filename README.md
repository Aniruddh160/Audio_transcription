This Code is used to transcribe audio that is posted to the FastAPI

The requiremenst are given in a txt file in order to install all the necessary requirements :  run : pip install -r requirements.txt : in you CLI

To run the Server (FastAPI) : run : uvicorn mani:app --reload : in your CLI

How to access the entire application:

1. Clone the repo onto your Local Machine
2. After doing the above open your CLI and run the VENV use the following command: ./venv/scripts/activate
3. Then use the Uvicorn command to activate the FastAPI server (uvicorn mani:app --reload).
4. After the application has started change the URL from 127.0.0.0/8000 to 127.0.0.0/8000/docs in order to access the FastAPI documentation
5. After opening the FastAPi documentation Navigate to Diarize endpoint and click on try it out.
6. Provide a '.wav' file to diarize and transcribe.
7. Wait for a few minutes and the ouput can be seen as a JSON below.
                                                        
For HUGGINGFACE token : The fix is go to HuggingFace and create a token for use it in the program by decclaring it as an environment variable:

$env:HF_TOKEN="paste the token here", use this command in powershell in CLI.
