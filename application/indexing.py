import sys, os
import json
import boto3
import botocore
import time
from pprint import pprint
from utils import bedrock, print_ww
from utils.bedrock import bedrock_info
import shutil
from glob import glob
import cv2
import math
import base64
import numpy as np
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace
from pdf2image import convert_from_path
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredAPIFileLoader
from langchain.schema import Document
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from utils.common_utils import to_pickle, load_pickle
from utils.common_utils import retry
from utils.common_utils import print_html
from utils.ssm import parameter_store
from utils.opensearch_summit import opensearch_utils
from utils.chunk import parant_documents
from itertools import chain

def add_python_path(module_path):
    if os.path.abspath(module_path) not in sys.path:
        sys.path.append(os.path.abspath(module_path))
        print(f"python path: {os.path.abspath(module_path)} is added")
    else:
        print(f"python path: {os.path.abspath(module_path)} already exists")
    print("sys.path: ", sys.path)

module_path = "../../.."
add_python_path(module_path)

############### functions ###############
def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size

def check_image_size(img_path):
    return all(dim >= 100 for dim in get_image_size(img_path))

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())      
    return encoded_string.decode('utf-8')

def summary_img(summarize_chain, img_base64):
    img = Image.open(BytesIO(base64.b64decode(img_base64)))

    ### 단순 출력하는 부분이므로 주석 처리
#     plt.imshow(img) 
#     plt.show()

    summary = summarize_chain.invoke(
        {
            "image_base64": img_base64
        }
    )
    return summary

def show_opensearch_doc_info(response):
    print("opensearch document id:" , response["_id"])
    print("family_tree:" , response["_source"]["metadata"]["family_tree"])
    print("parent document id:" , response["_source"]["metadata"]["parent_id"])
    print("parent document text: \n" , response["_source"]["text"])


############### main code ###############
def indexing():
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    llm_text = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-Sonnet"),
        client=boto3_bedrock,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "max_tokens": 2048,
            "stop_sequences": ["\n\nHuman"],
            # "temperature": 0,
            # "top_k": 350,
            # "top_p": 0.999
        }
    )
    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name="Titan-Embeddings-G1") #Titan-Text-Embeddings-V2
    )
    dimension = 1536 #1024
    print("Bedrock Embeddings Model Loaded")

    image_path = "./fig"

    file_path = "./data/complex_pdf/school_edu_guide.pdf"

    if os.path.isdir(image_path): shutil.rmtree(image_path)
    os.mkdir(image_path)

    loader = UnstructuredFileLoader(
        file_path=file_path,

        chunking_strategy = "by_title",
        mode="elements",

        strategy="hi_res",
        hi_res_model_name="yolox", #"detectron2_onnx", "yolox", "yolox_quantized"

        extract_images_in_pdf=True,
        #skip_infer_table_types='[]', # ['pdf', 'jpg', 'png', 'xls', 'xlsx', 'heic']
        pdf_infer_table_structure=True, ## enable to get table as html using tabletrasformer

        extract_image_block_output_dir=image_path,
        extract_image_block_to_payload=False, ## False: to save image

        max_characters=4096,
        new_after_n_chars=4000,
        combine_text_under_n_chars=2000,

        languages= ["kor+eng"],

        post_processors=[clean_bullets, clean_extra_whitespace]
    )

    docs = loader.load()

    to_pickle(docs, "./data/complex_pdf/pickle/parsed_unstructured.pkl")
    docs = load_pickle("./data/complex_pdf/pickle/parsed_unstructured.pkl")

    tables, texts = [], []
    images = glob(os.path.join(image_path, "*"))

    images = [img for img in images if check_image_size(img)]

    print(f"Large images (width, height >= 100): {len(images)}")

    tables, texts = [], []

    for doc in docs:

        category = doc.metadata["category"]

        if category == "Table": tables.append(doc)
            
        elif category == "Image": 
            print(f' {doc}')
            if check_image_size(doc) == True:
                images.append(doc)

        else: texts.append(doc)
        
        images = glob(os.path.join(image_path, "*"))

    print (f' # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}')
    table_as_image = True
    if table_as_image:
        image_tmp_path = os.path.join(image_path, "tmp")
        if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
        os.mkdir(image_tmp_path)
        
        # from pdf to image
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            print (f'pdf page {i}, size: {page.size}')    
            page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")

        print ("==")

        #table_images = []
        for idx, table in enumerate(tables):
            points = table.metadata["coordinates"]["points"]
            page_number = table.metadata["page_number"]
            layout_width, layout_height = table.metadata["coordinates"]["layout_width"], table.metadata["coordinates"]["layout_height"]

            img = cv2.imread(f'{image_tmp_path}/{page_number}.jpg')
            crop_img = img[math.ceil(points[0][1]):math.ceil(points[1][1]), \
                        math.ceil(points[0][0]):math.ceil(points[3][0])]
            table_image_path = f'{image_path}/table-{idx}.jpg'
            cv2.imwrite(table_image_path, crop_img)
            #table_images.append(table_image_path)

            print (f'unstructured width: {layout_width}, height: {layout_height}')
            print (f'page_number: {page_number}')
            print ("==")

            width, height, _ = crop_img.shape
            image_token = width*height/750
            print (f'image: {table_image_path}, shape: {img.shape}, image_token_for_claude3: {image_token}' )

            ## Resize image
            if image_token > 1500:
                resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                print("   - resize_img.shape = {0}".format(resize_img.shape))
                table_image_resize_path = table_image_path.replace(".jpg", "-resize.jpg")
                cv2.imwrite(table_image_resize_path, resize_img)
                os.remove(table_image_path)
                table_image_path = table_image_resize_path

            img_base64 = image_to_base64(table_image_path)
            table.metadata["image_base64"] = img_base64

        if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
        #print (f'table_images: {table_images}')
        images = glob(os.path.join(image_path, "*"))
        print (f'images: {images}')

    system_prompt = "You are an assistant tasked with describing table and image."
    system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_prompt = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + "{image_base64}",
            },
        },
        {
            "type": "text",
            "text": '''
                    Given image, give a concise summary.
                    Don't insert any XML tag such as <text> and </text> when answering.
                    Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )

    summarize_chain = prompt | llm_text | StrOutputParser()

    img_info = [image_to_base64(img_path) for img_path in images if os.path.basename(img_path).startswith("figure")]

    image_summaries = []
    for idx, img_base64 in enumerate(img_info):
        summary = summary_img(summarize_chain, img_base64)
        image_summaries.append(summary)
        print ("\n==")
        print (idx)

    # 요약된 내용을 Document의 page_content로, OCR결과는 metadata의 origin_image로 사용
    images_preprocessed = []

    for img_path, image_base64, summary in zip(images, img_info, image_summaries):
        
        metadata = {}
        metadata["img_path"] = img_path
        metadata["category"] = "Image"
        metadata["image_base64"] = image_base64
        
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        images_preprocessed.append(doc)
    
    ### Table 처리 ###
    human_prompt = [
        {
            "type": "text",
            "text": '''
                    Here is the table: <table>{table}</table>
                    Given table, give a concise summary.
                    Don't insert any XML tag such as <table> and </table> when answering.
                    Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )
    
    #summarize_chain = prompt | llm_text | StrOutputParser()
    summarize_chain = {"table": lambda x:x} | prompt | llm_text | StrOutputParser()

    # table_by_pymupdf = False
    # table_by_llama_parse = False

    # if table_by_llama_parse and table_by_pymupdf:
    #     tables = tables + docs_table_llamaparse + docs_table_pymupdf
    # elif table_by_llama_parse:
    #     tables = tables + docs_table_llamaparse
    # elif table_by_pymupdf:
    #     tables = tables + docs_table_pymupdf

    table_info = [(t.page_content, t.metadata["text_as_html"]) for t in tables]
    table_summaries = summarize_chain.batch(table_info, config={"max_concurrency": 1})
    if table_as_image: 
        table_info = [(t.page_content, t.metadata["text_as_html"], t.metadata["image_base64"]) if "image_base64" in t.metadata else (t.page_content, t.metadata["text_as_html"], None) for t in tables]
    verbose = True
    index = 0
    if verbose:
        for table, summary in zip(table_info, table_summaries):

            if table_as_image:
                page_contents, table_as_html, img_base64 = table
            else: page_contents, table_as_html = table

            print ("============================")
            print (index)
            print (f'table: {page_contents}')
            print ("----------------------------")
            print (f'summary: {summary}')
            print ("----------------------------")
            print (f'html:')
            print_html(table_as_html)
            print ("----------------------------")
            if table_as_image and img_base64 is not None:
                print ("image")
                img = Image.open(BytesIO(base64.b64decode(img_base64)))
                plt.imshow(img)
                plt.show()
            index += 1
    # 요약된 내용을 Document의 page_content로, parsed table은 metadata의 origin_table로 사용
    tables_preprocessed = []
    for origin, summary in zip(tables, table_summaries):
        metadata = origin.metadata
        metadata["origin_table"] = origin.page_content
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        tables_preprocessed.append(doc)
    
    ### Index 생성 ###
    region=boto3.Session().region_name
    pm = parameter_store(region)

    index_name = "default_doc_index"
    pm.put_params(
        key="opensearch_index_name",
        value=f'{index_name}',
        overwrite=True,
        enc=False
    )
    index_body = {
        'settings': {
            'analysis': {
                'analyzer': {
                    'my_analyzer': {
                            'char_filter':['html_strip'],
                        'tokenizer': 'nori',
                        'filter': [
                            #'nori_number',
                            #'lowercase',
                            #'trim',
                            'my_nori_part_of_speech'
                        ],
                        'type': 'custom'
                    }
                },
                'tokenizer': {
                    'nori': {
                        'decompound_mode': 'mixed',
                        'discard_punctuation': 'true',
                        'type': 'nori_tokenizer'
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "J", "XSV", "E", "IC","MAJ","NNB",
                            "SP", "SSC", "SSO",
                            "SC","SE","XSN","XSV",
                            "UNA","NA","VCP","VSV",
                            "VX"
                        ]
                    }
                }
            },
            'index': {
                'knn': True,
                'knn.space_type': 'cosinesimil'  # Example space type
            }
        },
        'mappings': {
            'properties': {
                'metadata': {
                    'properties': {
                        'source': {'type': 'keyword'},
                        'page_number': {'type':'long'},
                        'category': {'type':'text'},
                        'file_directory': {'type':'text'},
                        'last_modified': {'type': 'text'},
                        'type': {'type': 'keyword'},
                        'image_base64': {'type':'text'},
                        'origin_image': {'type':'text'},
                        'origin_table': {'type':'text'},
                    }
                },
                'text': {
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer',
                    'type': 'text'
                },
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': f"{dimension}" # Replace with your vector dimension
                }
            }
        }
    }
    opensearch_domain_endpoint = pm.get_params(
        key="opensearch_domain_endpoint",
        enc=False
    )

    opensearch_user_id = pm.get_params(
        key="opensearch_user_id",
        enc=False
    )
    secrets_manager = boto3.client('secretsmanager')

    response = secrets_manager.get_secret_value(
        SecretId='opensearch_user_password'
    )

    secrets_string = response.get('SecretString')
    secrets_dict = eval(secrets_string)
    #opensearch_user_password= list(secrets_dict.values())[0]

    opensearch_user_password = secrets_dict['pwkey']
    opensearch_domain_endpoint = opensearch_domain_endpoint
    rag_user_name = opensearch_user_id
    rag_user_password = opensearch_user_password

    http_auth = (rag_user_name, rag_user_password)

    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)

    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )
    index_exists = opensearch_utils.check_if_index_exists(
        os_client,
        index_name
    )

    if index_exists:
        opensearch_utils.delete_index(
            os_client,
            index_name
        )

    opensearch_utils.create_index(os_client, index_name, index_body)
    index_info = os_client.indices.get(index=index_name)
    print("Index is created")
    pprint(index_info)

    vector_db = OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=f"https://{opensearch_domain_endpoint}",
        embedding_function=llm_emb,
        http_auth=http_auth, # http_auth
        is_aoss=False,
        engine="faiss",
        space_type="l2",
        bulk_size=100000,
        timeout=60
    )

    parent_chunk_size = 4096
    parent_chunk_overlap = 0

    child_chunk_size = 1024
    child_chunk_overlap = 256

    opensearch_parent_key_name = "parent_id"
    opensearch_family_tree_key_name = "family_tree"

    parent_chunk_docs = parant_documents.create_parent_chunk(
        docs=texts,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap
    )
    print(f'Number of parent_chunk_docs= {len(parent_chunk_docs)}')


    parent_ids = vector_db.add_documents(
        documents = parent_chunk_docs, 
        vector_field = "vector_field",
        bulk_size = 1000000
    )
    total_count_docs = opensearch_utils.get_count(os_client, index_name)
    print("total count docs: ", total_count_docs)

    response = opensearch_utils.get_document(os_client, doc_id = parent_ids[0], index_name = index_name)
    show_opensearch_doc_info(response)
    child_chunk_docs = parant_documents.create_child_chunk(
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        docs=parent_chunk_docs,
        parent_ids_value=parent_ids,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name
    )

    print(f"Number of child_chunk_docs= {len(child_chunk_docs)}")
    parent_id = child_chunk_docs[0].metadata["parent_id"]
    print("child's parent_id: ", parent_id)
    print("\n###### Search parent in OpenSearch")
    response = opensearch_utils.get_document(os_client, doc_id = parent_id, index_name = index_name)
    show_opensearch_doc_info(response)    
    child_chunk_docs[0]

    for table in tables_preprocessed:
        table.metadata["family_tree"], table.metadata["parent_id"] = "parent_table", "NA"
    for image in images_preprocessed:
        image.metadata["family_tree"], image.metadata["parent_id"] = "parent_image", "NA"

    docs_preprocessed = list(chain(child_chunk_docs, tables_preprocessed, images_preprocessed))
    docs_preprocessed = list(chain(images_preprocessed))
    images_preprocessed[0]

    child_ids = vector_db.add_documents(
        documents=docs_preprocessed, 
        vector_field = "vector_field",
        bulk_size=1000000
    )
    print("length of child_ids: ", len(child_ids))

    print("\n\n\n\n\n\n======= DONE !!!!!! ======")


### RUN ###

if __name__ == "__main__":
	indexing()