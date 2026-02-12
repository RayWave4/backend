from fastapi import FastAPI, status 
from starlette.responses import RedirectResponse 
from routes import juniagpt 

app = FastAPI(title="JuniaGPT", version="1.0.0") 

# Inclusion du routeur que nous avons corrigé précédemment
app.include_router(juniagpt.router) 

@app.get( 
    "/", 
    tags=["startup"], 
    description="API startup on documentation page.", 
) 
def main(): 
    """Redirect to the api documentation at launch.""" 
    return RedirectResponse(url="/docs") 

@app.get( 
    "/healthcheck", 
    tags=["healthcheck"], 
    status_code=status.HTTP_200_OK, 
    response_description="ok", 
    summary="resume", 
) 
def get_api_status() -> str: 
    """Return Ok if the api is up.""" 
    return "ok"