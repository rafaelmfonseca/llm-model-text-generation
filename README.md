```py
python -m venv .env
.env/Scripts/activate

pip3 install transformers
pip3 install --upgrade huggingface_hub
pip3 install python-dotenv
pip3 install accelerate
pip3 install jsonformer
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip3 install -r requirements.txt
# pip3 freeze > requirements.txt

python .\meta_llama3_8B_instruct.py
```