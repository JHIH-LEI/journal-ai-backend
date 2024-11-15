from fastapi import FastAPI,File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os;
import tempfile;
import shutil;
import whisper
import uvicorn;
from langchain_core.prompts import PromptTemplate;
from langchain_core.output_parsers import PydanticOutputParser;
from langchain_openai import ChatOpenAI;
from pydantic import BaseModel, Field;
from typing import List;
from datetime import datetime;

app = FastAPI();

model = whisper.load_model("small")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_FOLDER = 'voiceToJournal/'

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

class Event(BaseModel):
    """event that happen in journal"""
    
    event_title: str = Field(description="event subject")
    event_index: int = Field(description="event index, indicate the order of event, start from 0, should be continue and unique")
    event_body: str = Field(description="event content, descript detail about what happend in this event")
    
    
class Journal(BaseModel):
    """journal which might include multiple event in that day"""
    
    journal_title: str = Field(description="journal title", default=datetime.now().strftime("%Y-%m-%d"))
    journal_body: str = Field(description="journal summary in that day")
    events: List[Event] = Field(description="events that happen in journal")


@app.post('/journals')
def createJournalWithVoice(file: UploadFile = File(...)):
    try :
        # 使用臨時資料夾來存儲文件
        with tempfile.NamedTemporaryFile(dir=TEMP_FOLDER, delete=False, suffix=".webm") as temp_file:
            # 將上傳的文件內容寫入臨時文件
           shutil.copyfileobj(file.file, temp_file)
           temp_file_path = temp_file.name
        
        result = model.transcribe(temp_file_path, fp16=False)
        
        # prompt -> llm model -> parser
        llm_model = ChatOpenAI(model="gpt-4o-mini",temperature=0, verbose=True)
        parser = PydanticOutputParser(pydantic_object=Journal)
        
        prompt_template = PromptTemplate(
            template="將以下文字整理成一篇日記 {journal} 並且找出發生的事件有哪些，依序做歸類，按照日記摘要、事件主題、事件內容來書寫. \n{format_instructions}\n",
            input_variables=['journal'],
            partial_variables={"format_instructions": parser.get_format_instructions()}
            )
        
        chain =  prompt_template| llm_model | parser
        final_result = chain.invoke({"journal": result['text']})


        return {"data": final_result, "success": True, "message": "success", "code": 200}
    
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"data": None, "success": False, "message": "error", "code": 500}
        # TODO: 更多的錯誤處理邏輯
    
    finally:
        # 處理完畢後刪除臨時文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)