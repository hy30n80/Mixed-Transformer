import base64
from io import BytesIO
from PIL import Image
import json
import matplotlib.pyplot as plt
import os
import random
import re
from datasets import load_dataset


def donwload_dataset():
    dataset = load_dataset("Phineas476/EmbSpatial-Bench", cache_dir='./')
    print(dataset)

    # 특정 데이터셋 샘플 확인 (예: 훈련 데이터의 첫 번째 항목)
    sample = dataset['train'][0]
    print(sample)

    # 데이터셋의 샘플을 반복하며 처리
    for i, data in enumerate(dataset['train']):
        print(f"Sample {i}: {data}")
        if i == 4:  # 첫 5개의 샘플만 출력
            break


def modify_dataset_architecture(input_file="embspatial_sft.json", image_folder="./Image/embspatial_sft", output_file="modified_sft.json"):
    # JSON 파일 열기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i in range(len(data)):
        base64_string = data[i]['image']
        image_data = base64.b64decode(base64_string)

        image = Image.open(BytesIO(image_data))
        
        file_name = 'sft_' + str(i) + '.jpg'
        image.save(os.path.join(image_folder, file_name), 'JPEG')
        print(f"Image saved : {i}/{len(data)}")
        data[i]['image'] = file_name

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def create_semantic_negated_caption(input_file, output_file):
    # JSON 파일 열기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    transformed_data = []

    for i in range(len(data)):
        if data[i]['relation'] == "right":
            caption = data[i]['answer']
            negated_caption = caption.replace("right", "left")
        
        elif data[i]['relation'] == "left":
            caption = data[i]['answer']
            negated_caption = caption.replace("left", "right")

        elif data[i]['relation'] == "close":
            answer_name = data[i]['answer']
            filtered_objects = [obj["name"] for obj in data[i]["objects"] if obj["name"] != answer_name]

            if filtered_objects:
                random_object = random.choice(filtered_objects)
            
            caption = f"The {answer_name} is closer than the {random_object}"
            negated_caption = f"The {answer_name} is farther than the {random_object}"
            

        elif data[i]['relation'] == "far":
            answer_name = data[i]['answer']
            filtered_objects = [obj["name"] for obj in data[i]["objects"] if obj["name"] != answer_name]

            if filtered_objects:
                random_object = random.choice(filtered_objects)
            
            caption = f"The {answer_name} is farther than the {random_object}"
            negated_caption = f"The {answer_name} is closer than the {random_object}"


        elif data[i]['relation'] == "under":
            caption = data[i]['answer']
            negated_caption = re.sub(r'\b(beneath|below|under)\b', 'above', caption)


        elif data[i]['relation'] == "above":
            caption = data[i]['answer']

            replacement_words = ["beneath", "below", "under"]
            random_word = random.choice(replacement_words)
            negated_caption = caption.replace("above", random_word)


        transformed_item = {
            "image" : data[i]["image"],
            "caption" : caption,
            "negated_caption" : negated_caption,
            "relation" : data[i]['relation'],
            "data_source": data[i]["data_source"]
        }

        transformed_data.append(transformed_item)

        print(f"Progress : {i}/{len(data)}")


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)


def swap_subject_object(sentence):
    # "the" 뒤에 있는 단어를 모두 찾음
    the_words = re.findall(r'[Tt]he\s+(\w+)', sentence)
    #caption = sentence

    if len(the_words) < 2:
        return sentence  # 만약 "the" 뒤의 단어가 2개 미만이면, 변경하지 않고 원래 문장 반환

    # 첫 번째와 마지막 단어 추출
    first_word = the_words[0]
    last_word = the_words[-1]

    # 첫 번째 단어를 마지막 단어로, 마지막 단어를 첫 번째 단어로 바꿈
    sentence = re.sub(f'{last_word}', f'{first_word}', sentence, count=1)
    sentence = re.sub(f'{first_word}', f'{last_word}', sentence, count=1)
    

    return sentence


def create_structural_negated_caption(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    transformed_data = []

    for i in range(len(data)):
        caption = data[i]['caption']
        structural_negated_caption = swap_subject_object(caption)

        transformed_item = {
            "image" : data[i]["image"],
            "caption" : caption,
            "structural_negated_caption" : structural_negated_caption,
            "semantic_negated_caption" : data[i]['negated_caption'],
            "relation" : data[i]['relation'],
            "data_source": data[i]["data_source"]
        }

        transformed_data.append(transformed_item)

        print(f"Progress : {i}/{len(data)}")


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)





def split_dataset(input_file="filtered_sft.json", train_file="filtered_sft/train.json", test_file="filtered_sft/test.json"):

    # JSON 파일 읽기
    with open(input_file, "r") as f:
        data = json.load(f)

    # 데이터를 무작위로 섞기
    random.shuffle(data)

    # 훈련 세트와 테스트 세트로 분할
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # 훈련 세트와 테스트 세트 저장
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"Training data saved to {train_file}")
    print(f"Test data saved to {test_file}")




if __name__=="__main__":
    # <Download Dataset>
    donwload_dataset()
    
    # <Modify Dataset Architecture>
    image_folder = "./Image/embspatial_sft"

    dataset = "embspatial_sft.json"
    modified_dataset = "modified_sft.json"
    filtered_dataset = "filtered_sft.json"

    # <Modify Dataset Architecture> - Convert Base64 to Image
    modify_dataset_architecture(dataset, image_folder, modified_dataset)

    # <Create Filtered Dataset> - Negated Caption Generation
    create_structural_negated_caption(modified_dataset, filtered_dataset)

    # <Split Dataset> - Split Dataset for Training and Testing
    split_dataset(filtered_dataset)

