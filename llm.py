# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import json
import time
import ast
from tqdm import tqdm
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Loading model...")
# Load the tokenizer and model from the merged checkpoint
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
    # attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    # device map to mps of mac
    device_map={"": device} if device.type != "cpu" else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,    
)
print("Model loaded.")

df = pd.read_json("metadata.json")
df["content"] = df["path"].apply(lambda p : Path(p).read_text(encoding="utf-8"))

with open("mna_definitions.json", "r") as f:
    mna_definitions = json.load(f)

print("Length of M&A definition:", len(mna_definitions))

def inference(mna_name, args, sentence):
    start_time = time.time()

    label = args['label']
    argument_labels = args['arguments']  # This is a dict {field: definition}

    print("M&A NAME:", mna_name)
    print("LABEL:", label)

    # Phase 1: Detect all spans indicating events of this type
    phase1_prompt = f"""
    Bạn là một hệ thống trích xuất thông tin từ văn bản.
    
    Link trang web: "{mna_name}"
    Nhãn: "{label}"
    
    M&A là các hoạt động kinh doanh trong đó một công ty hoặc tập đoàn thực hiện các thương vụ mua lại hoặc sáp nhập với một công ty hoặc tập đoàn khác để tạo ra một thực thể mới có quy mô lớn hơn và có thể mang lại lợi ích kinh tế cho các bên liên quan
    Nhiệm vụ của bạn là đọc đoạn văn bên dưới và xác định xem có bất kỳ mô tả rõ ràng nào về một quá trình M&A (Merge & Acquisitions) hay không. 
    
    Đoạn văn:
    "{sentence}"
    
    Hãy thực hiện các bước sau để phân tích:
    
    1. Xác định từ khóa hoặc cụm từ liên quan đến thương vụ M&A: Dựa trên định nghĩa của thương vụ "{mna_name}", liệt kê các từ khóa hoặc cụm từ trong đoạn văn có thể chỉ ra thương vụ M&A này.
    2. Kiểm tra ngữ cảnh: Xác định xem các từ khóa hoặc cụm từ đó có được sử dụng trong ngữ cảnh mô tả rõ ràng một thương vụ M&A cụ thể hay không. Một thương vụ M&A cụ thể cần có chi tiết như ai làm gì, ở đâu, hoặc khi nào.
    3. Lọc các mô tả rõ ràng: Chỉ chọn các đoạn văn bản mô tả rõ một thương vụ M&A thuộc loại "{mna_name}". Loại bỏ các đoạn không rõ ràng, suy diễn, hoặc không liên quan trực tiếp đến thương vụ M&A.
    4. Tạo đầu ra: 
       - Nếu văn bản không mô tả về một thương vụ M&A, trả về nhãn: `0`.
       - Nếu văn bản có mô tả về một thương vụ M&A, trả về nhãn: '1'

    OUTPUT:
    """

    messages = [{"role": "user", "content": phase1_prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=True, top_p=1, temperature=0.5, max_new_tokens=512)[:, inputs["input_ids"].shape[1]:]

    spans_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('[PHASE 1 DETECTED SPANS]', spans_output)

    elapsed = time.time() - start_time

    print(f"[LOG] Detect M&A website finished with {elapsed:.2f} seconds")

    # Phase 2: Extract structured arguments from each detected span
    # field_descriptions = "\n".join(
    #     [f'- "{field}": {definition}' for field, definition in argument_defs.items()]
    # )

    # elapsed = time.time() - start_time

    # print(f"[LOG] Event finished with {elapsed:.2f} seconds")

    # print("[FIELD DESCRIPTION]", field_descriptions)

    # spans_output = ast.literal_eval(spans_output)

    # for i, span in enumerate(spans_output):
    #     print(i, span)
    #     phase2_prompt = f"""
    #     Tên sự kiện: {path}
    #     Định nghĩa sự kiện: {event_def}
        
    #     Hãy trích xuất các trường dữ liệu sau từ đoạn văn:
    #     {field_descriptions}
        
    #     Đoạn văn:
    #     {span}
        
    #     Trích xuất các thông tin được yêu cầu trong trường dữ liệu từ đoạn văn.
    #     Nếu trong đoạn văn không đề cập thông tin đến trường nào, hãy điền "NULL".
        
    #     Đầu ra cần là một đối tượng JSON với các khóa tương ứng với tên trường, không kèm theo bất kỳ text nào khác. Trả lời bằng tiếng Việt, field tiếng Anh.
    #     """

    #     messages = [{"role": "user", "content": phase2_prompt}]
    #     inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

    #     with torch.no_grad():
    #         outputs = model.generate(inputs, do_sample=False, top_p=1, temperature=1, max_new_tokens=256)[:, inputs["input_ids"].shape[1]:]

    #     json_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print(f'[PHASE 2 - EVENT {i+1}]', json_response)
        
    #     results.append(json_response)

    return spans_output

def apply_inference_to_df(df):
    # Initialize a new column for extracted events
    df['predicted_label'] = [-1 for _ in range(len(df))]
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        start_time = time.time()
        content = row['content']
        mna_label = 0
        print('[CONTENT]', content)

        print('REFINENING...')
        refinement_prompt = f"""
        Bạn là một hệ thống chỉnh sửa văn bản, chuyên làm sạch các đoạn văn có lỗi định dạng hoặc nội dung không rõ ràng.
        
        Cho đoạn văn sau:
        {content}
        
        Nhiệm vụ của bạn là viết lại đoạn văn để làm sạch các lỗi định dạng và nội dung không rõ ràng, đồng thời giữ nguyên các câu văn bình thường, rõ ràng và đúng ngữ nghĩa. Hãy thực hiện các bước sau:
        
        1. Xác định các lỗi định dạng hoặc nội dung:
           - Tìm các câu hoặc đoạn bị lặp lại không cần thiết (ví dụ: cùng một câu xuất hiện nhiều lần).
           - Tìm các câu hoặc đoạn không hoàn chỉnh (ví dụ: câu bị cắt ngang hoặc thiếu thành phần chính).
           - Tìm các lỗi như dòng trống thừa, khoảng trắng không cần thiết, hoặc dấu câu sai.
        
        2. Xử lý các lỗi:
           - Loại bỏ các câu hoặc đoạn bị lặp lại, chỉ giữ lại lần xuất hiện đầu tiên.
           - Sửa các câu không hoàn chỉnh bằng cách hoàn thiện ngữ nghĩa nếu có đủ ngữ cảnh, hoặc loại bỏ nếu không thể sửa.
           - Xóa các dòng trống thừa hoặc chuẩn hóa dấu câu để văn bản mạch lạc.
        
        3. Giữ nguyên nội dung gốc:
           - Các câu hoặc đoạn văn rõ ràng, đúng ngữ nghĩa, và không có lỗi phải được giữ nguyên hoàn toàn.
        
        4. Tạo đầu ra:
           - Trả về đoạn văn đã được làm sạch, định dạng thành các đoạn văn bản mạch lạc, tự nhiên bằng tiếng Việt.
           - Không thêm giải thích hoặc nội dung mới ngoài việc làm sạch văn bản.

        OUTPUT:
        """

        messages = [{"role": "user", "content": refinement_prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=True, top_p=1, temperature=0.8, max_new_tokens=2048)[:, inputs["input_ids"].shape[1]:]

        refined_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('[REFINENED:]', refined_content)
        # Apply inference function
        for ma_dict in tqdm(mna_definitions, leave=False):
            mna_name = next(iter(ma_dict))
            args = ma_dict[mna_name]
            # print(mna_name, args)
            result = inference(mna_name, args, refined_content)
            if result == 1:
                mna_label = result
        
        # Store the collected events in the DataFrame
        df.at[idx, 'predicted_label'] = mna_label

        elapsed_time = time.time() - start_time
        print(f"Row took: {elapsed_time:.2f} seconds")
    
    return df

# Apply the function to the DataFrame
df = apply_inference_to_df(df)

# auto free mps
torch.device.empty_cache()