import streamlit as st
from pdfminer.high_level import extract_text
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text
from haystack import Document
from haystack.nodes import PreProcessor
from sentence_transformers import SentenceTransformer
import faiss
import openai



# Streamlit App

# Set your OpenAI API key
openai.api_key = "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"

 # Generate embeddings of snippets
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# PDF Processing and Indexing Functions
def process_pdf_and_create_index(pdf_file_path, index_path):
    global model
    # Extract text from PDF
    pdf_text = extract_text(pdf_file_path)
    btl_text= """ BLOOM TAXONOMY: Bloom's taxonomy is a hierarchical framework for
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
experiences.
"""
    # btl_text = """Below is the information on Blooms Taxonomy categories, there are 6 categories
    # Remember,Understand,Apply,Analyze,Evalvuate,Create categories, explaining below further on each of categories

    # Remember: keywords are 	list, recite, outline, define, name, match, quote, recall, identify, label, recognize.
    # Understand : Keywords are describe, explain, paraphrase, restate, give original examples of, summarize, contrast, interpret, discuss.
    # Apply : Keywords are calculate, predict, apply, solve, illustrate, use, demonstrate, determine, model, perform, present.
    # Analyze: keywords are classify, break down, categorize, analyze, diagram, illustrate, criticize, simplify, associate.
    # Evalvuate: keywords are choose, support, relate, determine, defend, judge, grade, compare, contrast, argue, justify, support, convince, select, evaluate.
    # Create: keywords are design, formulate, build, invent, create, compose, generate, derive, modify, develop.
    # """
    pdf_text += btl_text

    # Initialize a DocumentStore
    document_store = InMemoryDocumentStore()

    # Clean the text
    cleaned_text = clean_wiki_text(pdf_text)

    # Create Haystack Document snippets
    preprocessor = PreProcessor(split_length=300, split_overlap=1,
                                split_respect_sentence_boundary=True,
                                clean_empty_lines=True, clean_whitespace=True,
                                clean_header_footer=True)
    snippets = preprocessor.process(Document(content=cleaned_text))

    # Extract snippet content
    snippet_text = [snippet.content for snippet in snippets]

    # Generate embeddings of snippets
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(snippet_text)

    # Create FAISS vector index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return snippet_text

def process_and_query(index_path, snippet_text, query_text, k=5):
    global model
    index = faiss.read_index(index_path)
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embedding = model.encode([query_text])

    #query_embedding = model.encode([query_text])

    D, I = index.search(query_embedding, k)

    nearest_neighbors_data = []
    for idx in I[0]:
        nearest_neighbors_data.append(snippet_text[idx])

    return nearest_neighbors_data

def get_openai_resp(prompt):
    # using OpenAI's Completion module that helps execute
    # any tasks involving text

    # Then, you can call the "gpt-3.5-turbo" model
    model_engine = "gpt-3.5-turbo"

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

    if uploaded_pdf_file:
        index_path ='index_file.index'
        snippet_text = process_pdf_and_create_index(uploaded_pdf_file, index_path)
        k=5

        question = st.text_input("Enter your question")
        query_text = question + " What is bloom's taxonomy level -Remember,Understand,Apply,Analyze,Evalvuate and Create?"
        # query_text = question
        nearest_neighbors = process_and_query(index_path, snippet_text, query_text, k)
        knowledge_context = '\n'.join(nearest_neighbors)
        question = "GIVEN QUESTION : \n" + question
        knowledge_context = " RELEVENT KNOWLEDGE CONTEXT FROM CHAPTER : \n"+ knowledge_context
        context = """
        above is a question and knowledge context from chapter of 10th grade on our environment chapter ,
        please carefully assign it a blooms taxonomy category along with a confidence score,
        classify it into Remember,Understand,Apply,Analyze,Evalvuate,Create categories"""
        
        
        prompt= question+ knowledge_context + "\n"*3 + "\n"*3 + context





        if st.button("Answer"):
            response = get_openai_resp(prompt)
            st.text("AI Bot reply:")
            st.write(response)
            #st.write(knowledge_context)
            #st.write(query_text)

if __name__ == "__main__":
    main()