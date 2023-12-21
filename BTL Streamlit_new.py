import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import langchain
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.document_loaders import PyPDFDirectoryLoader
import fitz  # PyMuPDF
import PyPDF2
from langchain.llms import OpenAI



openai.api_key  = ""
embeddings=OpenAIEmbeddings(api_key="")
#Vector searchDb in pinecone
pinecone.init(api_key="",environment="gcp-starter")


text_to_add = """BLOOM TAXONOMY: Bloom's taxonomy is a hierarchical framework for
categorizing educational learning objectives. It was first published in 1956 by Benjamin
Bloom and a group of educators, and it has since been revised several times. The
taxonomy is divided into six levels of cognition, each of which represents a
progressively more complex level of thinking:
KnowledgeThe foundational level of Bloom's Taxonomy, Knowledge, involves the recall
of factual information and basic concepts. Learners demonstrate their ability to
remember and recognize previously learned material. This level is essential as it lays
the groundwork for higher-order thinking. Educators use verbs such as "define," "list,"
"identify," and "label" to design learning activities that reinforce students' understanding
of fundamental concepts.
Comprehension:Moving beyond memorization, the Comprehension level requires
learners to demonstrate their understanding of concepts by explaining ideas in their own
words. This involves interpreting and summarizing information, fostering deeper
comprehension. Verbs like "describe," "explain," "summarize," and "interpret" guide
teachers in crafting activities that promote critical thinking and interpretation.
Application:At the Application level, students are challenged to apply their knowledge
and understanding in real-world scenarios. They must use the information they have
learned to solve problems, complete tasks, and make connections to practical
situations. This level encourages practical application and transfer of knowledge. Verbs
such as "apply," "use," "solve," and "demonstrate" guide educators in creating hands-on
experiences for learners.
Analysis:The Analysis level requires learners to break down complex ideas into their
component parts, examine relationships, and draw conclusions based on given
information. Analytical thinking enhances students' ability to explore and evaluate
concepts critically. Verbs like "analyze," "compare," "contrast," and "differentiate" inform
educators in developing activities that foster analytical skills
Synthesis:Synthesis, the level of creative thinking, challenges learners to combine
elements in novel ways to create something new. By utilizing their previous knowledge
and skills, students generate innovative ideas, designs, or solutions. This level fosters
creativity and ingenuity in problem-solving. Verbs such as "create," "design," "compose,"
and "develop" guide teachers in designing open-ended projects that stimulate creativity.
Evaluation:At the pinnacle of Bloom's Taxonomy is the Evaluation level, where learners
make judgments and form opinions based on specific criteria. They critically assess
information, theories, or arguments and defend their viewpoints. This level encourages
students to think critically and make informed decisions. Verbs like "evaluate," "assess,"
"judge," and "justify" inspire educators to create activities that develop higher-order
thinking skills.The levels of Bloom's taxonomy are hierarchical, meaning that each level
builds on the previous level. For example, in order to apply information, you must first
understand it. And in order to evaluate information, you must first understand and
analyze it.Bloom's taxonomy can be used to help teachers design learning activities and
assessments. By identifying the level of cognition that they want students to achieve,
teachers can create activities that are appropriate for the level of thinking. For example,
if a teacher wants students to be able to apply information, they would create an activity
that requires students to use information to solve a problem or complete a task.Bloom's
taxonomy can also be used to help students learn. By understanding the different levels
of cognition, students can learn how to think critically and solve problems. They can
also learn how to identify the level of thinking that is required for different tasks.Bloom's
taxonomy is a valuable tool for understanding and assessing student learning. It can
help teachers design effective learning activities and assessments, and it can help
students learn how to think critically and solve problems.
ConclusionBloom's Taxonomy continues to be a valuable tool
in education, guiding
teachers in designing learning experiences that promote progressive cognitive
development. By understanding the different levels of cognition, educators can create
activities that challenge and inspire students to think critically, problem-solve, and
become independent learners. The taxonomy's revised version, including metacognition
and self-regulation, emphasizes the importance of reflective learning and self-directed
growth. By incorporating Bloom's Taxonomy into educational practices, both teachers
and students can embark on a journey of meaningful and transformative learning
experiences."""  


question_context="""above is a question and knowledge context from chapter of 10th grade on our environment chapter ,
         please carefully assign it a blooms taxonomy category along with a confidence score,
         classify it into Remember,Understand,Apply,Analyze,Evalvuate,Create categories"""








def add_text_to_pdf(input_pdf_path, newdir,newfilename, text_to_add):
    # Open the existing PDF file
    file_path_new = os.path.join(newdir, newfilename)
    pdf_document = fitz.open(input_pdf_path)
    page = pdf_document.new_page(width=500, height=800)
    text_annotation = page.insert_text((30, 30), text_to_add,fontsize=9)
    pdf_document.save(file_path_new)
    pdf_document.close()
    os.remove(input_pdf_path)
    return file_path_new

def save_pdf(dirname,filename):
    os.makedirs(dirname, exist_ok=True) 
    file_path = os.path.join(dirname,filename.name)
    with open(file_path, 'wb') as file:
            file.write(filename.read())
            st.success(f"File saved to: {file_path}") 
            return file_path    

def chunk_embedding(directory,index_name):
        #reading directory and chunking pdf file
        dir_loader=PyPDFDirectoryLoader(directory)   
        documents=dir_loader.load()   
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=text_splitter.split_documents(documents) 

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(        
            name=index_name,
            dimension=1536,
            metric='cosine')

        index=Pinecone.from_documents(chunks,embeddings,index_name=index_name)
        return index

def  process_query(question,index_name):  
        pinecone.init(api_key="",environment="gcp-starter")
        index = pinecone.Index(index_name)
        query_text = question + " What is bloom's taxonomy level -Remember,Understand,Apply,Analyze,Evalvuate and Create?" 
        query_embedd = embeddings.embed_query(query_text)
        res = index.query(query_embedd, top_k=5, include_metadata=True)
        matching_results = [x['metadata']['text'] for x in res['matches']] 
        #print(matching_results[0])
        return matching_results[0]

def get_final_answer(question,queryResult):
     matching_results =" RELEVENT KNOWLEDGE CONTEXT FROM CHAPTER as mentioned below: \n"+ queryResult
     prompt= question+ matching_results + "\n"*3 + "\n"*3 + question_context
     prompt=prompt+"\n"+"If the information provided does not contain the answer,mention do not have the answer from the context provided"
     if st.button("Answer"):
             response = get_openai_resp(prompt)
             st.text("ChatGPT API reply:")
             st.write(response)






def get_openai_resp(prompt):
    # using OpenAI's Completion module that helps execute
    # any tasks involving text
    # Then, you can call the "gpt-3.5-turbo" model
    model_engine = "gpt-3.5-turbo"
    openai.api_key  = "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"
    response = openai.ChatCompletion.create(
        model=model_engine,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}])
    output_text = response['choices'][0]['message']['content']
    print("ChatGPT API reply:", output_text)
    return output_text


         
def main():
    st.title("Bloom's Taxonomy Prediction")   
    uploaded_pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf_file is not None:
        #Saving uploaded pdf into directory uploads
        file_path_original= save_pdf("uploads",uploaded_pdf_file)

        #Adding Bloom texanomy information to pdf and creating new pdf with name modified.pdf
        file_path_modified=add_text_to_pdf(file_path_original,"uploads","modified.pdf", text_to_add)

      
        #chunking pdf,making embedding and upserting in pinecone db
        index=chunk_embedding("uploads","semanticsearch")

        question = st.text_input("Enter your question")
        
        if question:
           queryResult=process_query(question,'semanticsearch')
           get_final_answer(question,queryResult)
 
          





if __name__ == "__main__":
    main()
 


