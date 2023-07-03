# fastapi imports here
from fastapi import FastAPI, Request
from fastapi.response import JSONResponse

from pydantic import BaseModel
import deepchem as dc


# General classes
# Define the input data model 
class MoleculeInput(BaseModel):
    """
    Check that the molecule input we have is a legitimate
    string
    """

    smiles: str

# Exception classes
class SmilesException(Exception):
    """
    part of the custom exception for the molecule when
    we are not getting the right type of molecule 
    """
    def __init__(self, molecule_name: str):
        self.molecule_name = molecule_name

# custom exception handlers
@app.exception_handler(MoleculeException)
async def molecule_exception_handler(request: Request, exc: SmilesException):
    """
    """
    return JSONResponse(
        status_code = 404,
        content = {"message" : f"{exc.molecule_name} is not a valid smile!"}
    )

# -- get functions -- 
@app.get("/smiles/{smiles}")
def read_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise SmilesException(smiles = smiles)
    # not sure if this is correct..
    return {"mol": mol}

"
# Here we would need to load up the model

# Load the pre-trained DeepChem model
model = dc.models.GraphConvModel(n_tasks=1, mode="regression")
model.restore_from_dir("/path/to/pretrained/model/")

# Create the FastAPI app
app = FastAPI()

# Define the prediction route
@app.post("/predict")
def predict_molecule_properties(molecule: MoleculeInput):
    """
    post the predicted properities of the molecule on the predict link

    """
    smiles = molecule.smiles

    # Convert SMILES to molecular representation
    featurizer = dc.feat.ConvMolFeaturizer()
    mol = featurizer.featurize([smiles])[0]
    # if we dont manage to get the right mol value here
    if not mol:

        
    # Generate predictions
    _, y_pred = model.predict_on_batch([mol])

    return {"predicted_properties": y_pred.flatten().tolist()}


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
