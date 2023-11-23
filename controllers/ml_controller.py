from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output


from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'model.h5'

# Load the tokenizer from the file
tokenizer_filename = TOKENIZER_PATH
with open(tokenizer_filename, 'rb') as f:
    tokenizer = pickle.load(f)

# Load Tensorflow model
model = keras.models.load_model(MODEL_PATH)
print(model.summary())


router = APIRouter()

preds = []

@router.get('/ml')
def get_preds():
    return preds

@router.post('/ml', status_code=status.HTTP_201_CREATED, response_model=Prediction_Output)
def predict(pred_input: Prediction_Input):

    sequences = tokenizer.texts_to_sequences(pred_input.text_input)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    prediction_f = model.predict(padded)

    prediction_dict = {"id": str(pred_input.id), "text_input":str(pred_input.text_input) , "pred": float(prediction_f[0,0])}
    preds.append(prediction_dict)

    return prediction_dict

@router.delete('/ml', status_code=status.HTTP_200_OK)
def delete_preds():
    preds.clear()
    return preds

@router.put('/ml/{pred_id}', status_code=status.HTTP_202_ACCEPTED, response_model=Prediction_Output)
def update_pred(pred_id:int, pred_input: Prediction_Input):
    for pred in preds:
        if pred['id'] == pred_id:
            pred['text_input'] = pred_input.text_input
            sequences = tokenizer.texts_to_sequences(pred_input.text_input)
            padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            prediction_f = model.predict(padded)
            prediction_dict = {"id": str(pred_input.id), "text_input":str(pred_input.text_input) , "pred": float(prediction_f[0,0])}
            preds.append(prediction_dict)
            return prediction_dict

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Prediction with id {pred_input.id} not found')
