from fastapi import FastAPI
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# Initialize FastAPI application
app = FastAPI()

# Define a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from local FastAPI!"}

# Set your personal ngrok auth token to enable ngrok tunneling
ngrok.set_auth_token("2yi4gWhKiFYPsDR1MpD9n8Rg5so_6Hv3dyUMYC2Vmkho5UnX4")

# Open a public tunnel to port 8000
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Allow nested async loops (needed for Jupyter and Colab environments)
nest_asyncio.apply()

# Start the FastAPI server using Uvicorn
uvicorn.run(app, port=8000)