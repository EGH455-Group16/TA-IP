git clone https://github.com/EGH455-Group16/TA-IP.git
cd TA-IP
git checkout AQSA_MERGE
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main_api.py -m best3_openvino_2022.1_6shave.blob -c best3.json
