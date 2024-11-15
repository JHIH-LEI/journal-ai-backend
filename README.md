# Features

This project includes several features power bt AI.

## 1. Voice-based Journal Creation
Support creating journal entries through user voice input.

### Steps:
1. **Voice-to-Text Conversion (Whisper)**  
   Convert the user's voice input into text using Whisper.
   
2. **Text Processing with ChatGPT (LLM)**  
   Use ChatGPT to process the transcribed text and generate meaningful content based on a prompt.

3. **Output Data Structure Enforcement (Pydantic)**  
   Use Pydantic to validate and enforce the predefined output structure, ensuring data is formatted correctly for the frontend and database.
