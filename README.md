# MCP-Powered-PDF-Retrieval-Augmented-Generation-Assistant
In this we implement RAG Architecture using MCP
for quick start

### step 1:
- Install all the required packages
  ``` pip install -r requirements.txt```

### step 2:
-Run the mcp server 
```python mcp_rag.py```

then the server is activated as 
<img width="1701" height="1088" alt="Screenshot 2025-08-11 215252" src="https://github.com/user-attachments/assets/3ed02c8d-1b57-4411-a0fd-978ab6e65603" />

-After that use MCP Inspector in vscode to connect to the server URL to view the response
<img width="1886" height="1123" alt="Screenshot 2025-08-11 220024" src="https://github.com/user-attachments/assets/5a4da494-029a-44f8-9fba-9ce548630e23" />

-when we run the index_pdf_file MCP tool the entire given pdf will be converted into number of chunks with embeddings

<img width="1986" height="91" alt="Screenshot 2025-08-11 220806" src="https://github.com/user-attachments/assets/c807e4c9-8d7f-4479-8e8f-02b339e4150c" />

-when we run the second mcp tool rag_query with your question it will provide the answer from the pdf 

<img width="1274" height="370" alt="Screenshot 2025-08-11 220923" src="https://github.com/user-attachments/assets/cf9d8e1b-acdc-42ff-add1-556f07b1af0a" />


The response from the rag_query mcp tool is as 
#### Response

```{
  "success": true,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "The document is about the challenges and advancements in generating question-answer (QA) pairs from text, specifically in the context of Natural Language Processing (NLP) and machine learning models. It discusses the difficulties of data collection, text length, and multilingual handling, but also highlights the achievements of a model that can work on different domains and handle multilingual text with an accuracy of 72%. The document also mentions the use of Prompt Engineering based on pipelining to improve the accuracy and performance of the model, and the evaluation of its performance using the BLEU score. \n\nIn summary, the document is about the development and evaluation of a model for generating QA pairs from text using NLP and machine learning techniques. \n\nAnswer to the user question: The document is about the development and evaluation of a model for generating QA pairs from text using NLP and machine learning techniques."
      }
    ],
    "isError": false,
    "structuredContent": {
      "result": "The document is about the challenges and advancements in generating question-answer (QA) pairs from text, specifically in the context of Natural Language Processing (NLP) and machine learning models. It discusses the difficulties of data collection, text length, and multilingual handling, but also highlights the achievements of a model that can work on different domains and handle multilingual text with an accuracy of 72%. The document also mentions the use of Prompt Engineering based on pipelining to improve the accuracy and performance of the model, and the evaluation of its performance using the BLEU score. \n\nIn summary, the document is about the development and evaluation of a model for generating QA pairs from text using NLP and machine learning techniques. \n\nAnswer to the user question: The document is about the development and evaluation of a model for generating QA pairs from text using NLP and machine learning techniques."
    }
  }
}
```
